<objective>
Create an InfluxDB v2 docker-compose stack on the production server (ahs@blu) for time-series metrics storage.

This is part of a monitoring infrastructure to track bluetti-monitor metrics (battery percentage, camera connectivity, etc.) and enable Grafana alerting.
</objective>

<context>
Server: ssh ahs@blu (10.0.0.142)
Target directory: /home/ahs/influxdb
Port: 8086 (standard InfluxDB port, confirmed available)
Data persistence: ./instance/ folder for all persistent data

The user will later:
- Configure Nginx for HTTPS exposure
- Set up authentication
- Connect Grafana to this InfluxDB instance
</context>

<requirements>
1. SSH to the server and create directory structure:
   - /home/ahs/influxdb/
   - /home/ahs/influxdb/instance/ (for persistent data)
   - /home/ahs/influxdb/.gitignore with:
     ```
     instance/
     .env
     token.txt
     ```

2. Create docker-compose.yml with:
   - InfluxDB v2 (latest stable)
   - Port 8086 exposed
   - Volume mount: ./instance:/var/lib/influxdb2
   - Environment variables for initial setup (org, bucket, admin token)
   - Container name: influxdb
   - Restart policy: unless-stopped

3. Create .env file with sensible defaults:
   - DOCKER_INFLUXDB_INIT_MODE=setup
   - DOCKER_INFLUXDB_INIT_USERNAME=admin
   - DOCKER_INFLUXDB_INIT_PASSWORD=(generate a random 24-char password)
   - DOCKER_INFLUXDB_INIT_ORG=home
   - DOCKER_INFLUXDB_INIT_BUCKET=bluetti
   - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=(generate a random 48-char token)

4. Start the stack and verify it's running:
   - docker compose up -d
   - docker compose ps (should show healthy)
   - Verify port 8086 is listening

5. Output the credentials clearly so the user can save them
</requirements>

<implementation>
Use SSH commands to create all files directly on the server.
Generate secure random strings for password and token using: openssl rand -base64 36 | tr -dc 'a-zA-Z0-9' | head -c N

The docker-compose.yml should be minimal and follow best practices:
- Use named volumes or bind mounts to ./instance
- Don't hardcode secrets in compose file, use .env
- Include health check
</implementation>

<output>
All files created on the remote server via SSH.
Display the generated credentials at the end for the user to save.

IMPORTANT: Also save the token to /home/ahs/influxdb/token.txt so prompt 003 can read it:
```bash
grep DOCKER_INFLUXDB_INIT_ADMIN_TOKEN /home/ahs/influxdb/.env | cut -d= -f2 > /home/ahs/influxdb/token.txt
```
</output>

<verification>
Before declaring complete:
- Confirm container is running: docker compose ps shows "Up"
- Confirm port is accessible: curl http://localhost:8086/health returns JSON with status "pass"
- Display the admin credentials for user to record
</verification>

<success_criteria>
- InfluxDB container running on port 8086
- Data persisted to ./instance/ folder
- Credentials displayed for user to save
- Health check passes
</success_criteria>
