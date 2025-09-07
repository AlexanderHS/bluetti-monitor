# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based monitoring agent that captures images from an ESP32 webcam pointed at a Bluetti solar generator's LCD screen, extracts battery status using OCR/computer vision, and exposes the data via FastAPI endpoints.

## Architecture

The system follows this flow:
```
ESP32 Webcam → Bluetti Monitor Agent → FastAPI → Home Automation System
```

Key components:
- **Image Capture**: Periodic fetching from ESP32 webcam
- **OCR Processing**: Extract battery data from LCD screen images using Tesseract
- **REST API**: FastAPI endpoints for status, health, and metrics
- **Containerization**: Docker-based deployment

## Development Commands

### Local Development
```bash
# Initial setup
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with ESP32 webcam IP and settings

# Run development server
python main.py
```

### Docker Deployment
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f
```

## Configuration

Environment variables are configured in `.env`:
- `WEBCAM_URL`: ESP32 webcam base URL (required)
- `CAPTURE_INTERVAL`: Seconds between image captures (default: 30)
- `API_PORT`: FastAPI server port (default: 8000)
- `OCR_CONFIDENCE_THRESHOLD`: Minimum OCR confidence (default: 0.8)
- `LOG_LEVEL`: Logging verbosity (default: INFO)

## API Structure

The FastAPI application exposes:
- `GET /status`: Current battery status with percentage, timestamp, and OCR confidence
- `GET /health`: Service health check
- `GET /metrics`: Prometheus-style metrics (optional)

## Technology Stack

- **FastAPI**: REST API framework
- **OpenCV**: Image processing
- **Tesseract OCR**: Text extraction from LCD screen images
- **Docker**: Containerization
- **Python 3.11+**: Runtime environment

## Related Projects

This is part of a home automation system alongside:
- ESP32 Webcam hardware for LCD capture
- Smart Switch Control (planned) for charging automation