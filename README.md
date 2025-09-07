# Bluetti Monitor

**Author:** Alex Hamilton Smith  
**Project Type:** Personal Home Automation - Monitoring Agent  

## Overview

A Python-based monitoring agent that periodically captures images from an ESP32 webcam pointed at a Bluetti solar generator's LCD screen, parses battery status using OCR/computer vision, and exposes the data via FastAPI endpoints.

## Features

- **Periodic Image Capture**: Fetches images from ESP32 webcam at configurable intervals
- **OCR/Computer Vision**: Extracts battery percentage and status from LCD screen images  
- **REST API**: FastAPI endpoints providing JSON data on battery status
- **Docker Deployment**: Containerized for easy deployment and scaling
- **Status Tracking**: Monitors last seen battery level and timestamp

## API Endpoints

- `GET /status` - Current Bluetti battery status and metadata
- `GET /health` - Service health check
- `GET /metrics` - Prometheus-style metrics (optional)

### Example Response
```json
{
  "battery_percentage": 78,
  "last_updated": "2024-01-15T14:30:45Z",
  "seconds_ago": 12,
  "status": "charging",
  "image_capture_success": true,
  "ocr_confidence": 0.95
}
```

## Architecture

```
ESP32 Webcam → Bluetti Monitor Agent → FastAPI → Home Automation System
     |              |                    |              |
   Captures       Processes           Exposes        Consumes
   LCD Screen     with OCR            JSON API       Status Data
```

## Related Projects

- **[ESP32 Webcam](https://github.com/AlexanderHS/ESP32-webcam_Bluetti-Solar-Generator-Monitor)** - Camera hardware for LCD screen capture
- **Smart Switch Control** (TBD) - Automation logic for charging/output control

## Development

### Local Setup
```bash
# Clone and setup
git clone https://github.com/AlexanderHS/bluetti-monitor.git
cd bluetti-monitor
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your ESP32 webcam IP

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

Set environment variables in `.env`:
- `WEBCAM_URL` - ESP32 webcam base URL
- `CAPTURE_INTERVAL` - Seconds between captures (default: 30)
- `API_PORT` - FastAPI server port (default: 8000)

## Technology Stack

- **FastAPI** - REST API framework
- **OpenCV** - Image processing
- **Tesseract OCR** - Text extraction from images
- **Docker** - Containerization
- **Python 3.11+** - Runtime

---

*Part of a comprehensive home automation system for Bluetti solar generator management.*
