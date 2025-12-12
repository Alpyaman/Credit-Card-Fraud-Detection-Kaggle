# AWS ECS Task Definition for Fraud Detection API

resource "aws_ecs_cluster" "fraud_detection" {
  name = "fraud-detection-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Environment = "production"
    Project     = "fraud-detection"
  }
}

resource "aws_ecs_task_definition" "fraud_detection_api" {
  family                   = "fraud-detection-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name      = "fraud-detection-api"
      image     = "${aws_ecr_repository.fraud_detection.repository_url}:latest"
      essential = true
      
      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "MODEL_PATH"
          value = "/app/models/fraud_model.pkl"
        },
        {
          name  = "PREPROCESSOR_PATH"
          value = "/app/models/preprocessor.pkl"
        },
        {
          name  = "LOG_LEVEL"
          value = "INFO"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/fraud-detection-api"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

resource "aws_ecs_service" "fraud_detection_api" {
  name            = "fraud-detection-service"
  cluster         = aws_ecs_cluster.fraud_detection.id
  task_definition = aws_ecs_task_definition.fraud_detection_api.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.fraud_detection_api.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.fraud_detection_api.arn
    container_name   = "fraud-detection-api"
    container_port   = 8000
  }

  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 100
  }

  enable_ecs_managed_tags = true
  propagate_tags          = "SERVICE"

  tags = {
    Environment = "production"
    Project     = "fraud-detection"
  }
}

# ECR Repository
resource "aws_ecr_repository" "fraud_detection" {
  name                 = "fraud-detection-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Project = "fraud-detection"
  }
}

# Application Load Balancer
resource "aws_lb" "fraud_detection" {
  name               = "fraud-detection-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.public_subnet_ids

  enable_deletion_protection = true
  enable_http2               = true

  tags = {
    Environment = "production"
    Project     = "fraud-detection"
  }
}

resource "aws_lb_target_group" "fraud_detection_api" {
  name        = "fraud-detection-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
  }

  deregistration_delay = 30

  tags = {
    Project = "fraud-detection"
  }
}

resource "aws_lb_listener" "fraud_detection_https" {
  load_balancer_arn = aws_lb.fraud_detection.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = var.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.fraud_detection_api.arn
  }
}

# Security Groups
resource "aws_security_group" "alb" {
  name        = "fraud-detection-alb-sg"
  description = "Security group for fraud detection ALB"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "fraud-detection-alb-sg"
    Project = "fraud-detection"
  }
}

resource "aws_security_group" "fraud_detection_api" {
  name        = "fraud-detection-api-sg"
  description = "Security group for fraud detection API"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "fraud-detection-api-sg"
    Project = "fraud-detection"
  }
}

# IAM Roles
resource "aws_iam_role" "ecs_execution_role" {
  name = "fraud-detection-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_role_policy" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task_role" {
  name = "fraud-detection-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "fraud_detection_api" {
  name              = "/ecs/fraud-detection-api"
  retention_in_days = 30

  tags = {
    Project = "fraud-detection"
  }
}

# Auto Scaling
resource "aws_appautoscaling_target" "fraud_detection_api" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.fraud_detection.name}/${aws_ecs_service.fraud_detection_api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "fraud_detection_api_cpu" {
  name               = "fraud-detection-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.fraud_detection_api.resource_id
  scalable_dimension = aws_appautoscaling_target.fraud_detection_api.scalable_dimension
  service_namespace  = aws_appautoscaling_target.fraud_detection_api.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}

# Outputs
output "alb_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.fraud_detection.dns_name
}

output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.fraud_detection.repository_url
}
