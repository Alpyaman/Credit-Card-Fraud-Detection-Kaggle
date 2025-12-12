# Google Cloud Platform - Cloud Run Deployment

resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}

# Artifact Registry for Docker images
resource "google_artifact_registry_repository" "fraud_detection" {
  location      = var.region
  repository_id = "fraud-detection"
  description   = "Docker repository for fraud detection API"
  format        = "DOCKER"

  labels = {
    project = "fraud-detection"
  }
}

# Cloud Run Service
resource "google_cloud_run_service" "fraud_detection_api" {
  name     = "fraud-detection-api"
  location = var.region

  template {
    spec {
      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/fraud-detection/fraud-detection-api:latest"
        
        ports {
          container_port = 8000
        }

        resources {
          limits = {
            cpu    = "2000m"
            memory = "2Gi"
          }
        }

        env {
          name  = "MODEL_PATH"
          value = "/app/models/fraud_model.pkl"
        }

        env {
          name  = "PREPROCESSOR_PATH"
          value = "/app/models/preprocessor.pkl"
        }

        env {
          name  = "LOG_LEVEL"
          value = "INFO"
        }

        liveness_probe {
          http_get {
            path = "/health"
            port = 8000
          }
          initial_delay_seconds = 30
          period_seconds        = 10
          timeout_seconds       = 5
          failure_threshold     = 3
        }
      }

      container_concurrency = 80
      timeout_seconds       = 300

      service_account_name = google_service_account.fraud_detection_api.email
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale"      = "1"
        "autoscaling.knative.dev/maxScale"      = "10"
        "run.googleapis.com/cpu-throttling"     = "false"
        "run.googleapis.com/startup-cpu-boost"  = "true"
      }

      labels = {
        project     = "fraud-detection"
        environment = "production"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  autogenerate_revision_name = true

  depends_on = [google_project_service.required_apis]
}

# IAM policy to allow public access (adjust as needed)
resource "google_cloud_run_service_iam_member" "public_access" {
  service  = google_cloud_run_service.fraud_detection_api.name
  location = google_cloud_run_service.fraud_detection_api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Service Account for Cloud Run
resource "google_service_account" "fraud_detection_api" {
  account_id   = "fraud-detection-api"
  display_name = "Fraud Detection API Service Account"
  description  = "Service account for fraud detection API Cloud Run service"
}

# Cloud Storage bucket for models
resource "google_storage_bucket" "model_artifacts" {
  name          = "${var.project_id}-fraud-detection-models"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 5
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    project = "fraud-detection"
  }
}

# Grant service account access to model bucket
resource "google_storage_bucket_iam_member" "model_access" {
  bucket = google_storage_bucket.model_artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.fraud_detection_api.email}"
}

# Cloud Scheduler for model retraining
resource "google_cloud_scheduler_job" "retrain_model" {
  name        = "fraud-detection-retrain"
  description = "Trigger model retraining job"
  schedule    = "0 2 * * *" # Daily at 2 AM
  time_zone   = "UTC"

  http_target {
    http_method = "POST"
    uri         = "${google_cloud_run_service.fraud_detection_api.status[0].url}/retrain"

    oidc_token {
      service_account_email = google_service_account.fraud_detection_api.email
    }
  }

  retry_config {
    retry_count = 3
  }
}

# Cloud Monitoring - Uptime Check
resource "google_monitoring_uptime_check_config" "fraud_detection_api" {
  display_name = "Fraud Detection API Health Check"
  timeout      = "10s"
  period       = "60s"

  http_check {
    path         = "/health"
    port         = "443"
    use_ssl      = true
    validate_ssl = true
  }

  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = var.project_id
      host       = replace(google_cloud_run_service.fraud_detection_api.status[0].url, "https://", "")
    }
  }
}

# Alert Policy for API downtime
resource "google_monitoring_alert_policy" "api_downtime" {
  display_name = "Fraud Detection API Downtime"
  combiner     = "OR"

  conditions {
    display_name = "Uptime check failure"

    condition_threshold {
      filter          = "metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\" AND resource.type=\"uptime_url\""
      duration        = "300s"
      comparison      = "COMPARISON_LT"
      threshold_value = 1

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_FRACTION_TRUE"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]

  alert_strategy {
    auto_close = "1800s"
  }
}

# Notification Channel
resource "google_monitoring_notification_channel" "email" {
  display_name = "Fraud Detection Alerts"
  type         = "email"
  labels = {
    email_address = var.alert_email
  }
}

# Cloud Build trigger for CI/CD
resource "google_cloudbuild_trigger" "deploy_trigger" {
  name        = "fraud-detection-deploy"
  description = "Deploy fraud detection API on push to main"

  github {
    owner = var.github_owner
    name  = var.github_repo
    push {
      branch = "^main$"
    }
  }

  filename = "cloudbuild.yaml"
}

# Outputs
output "cloud_run_url" {
  description = "URL of the Cloud Run service"
  value       = google_cloud_run_service.fraud_detection_api.status[0].url
}

output "model_bucket_name" {
  description = "Name of the GCS bucket for models"
  value       = google_storage_bucket.model_artifacts.name
}

output "artifact_registry_url" {
  description = "URL of the Artifact Registry"
  value       = google_artifact_registry_repository.fraud_detection.name
}
