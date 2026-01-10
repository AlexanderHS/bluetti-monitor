<objective>
Create a Grafana docker-compose stack on the production server (ahs@blu) for dashboards and alerting.

This will connect to the InfluxDB instance (set up in previous prompt) to visualize bluetti-monitor metrics and send alerts when the ESP32 camera goes offline.
</objective>

<context>
Server: ssh ahs@blu (10.0.0.142)
Target directory: /home/ahs/grafana
Port: 3000 (standard Grafana port, confirmed available)
Data persistence: ./instance/ folder for all persistent data

InfluxDB is running at: http://localhost:8086
- Organization: home
- Bucket: bluetti
- Token: (user has this from previous setup)

The user will later:
- Configure Nginx for HTTPS exposure
- Set up Grafana authentication
- Create dashboards and alert rules manually
- Configure SMTP for email alerts
</context>

<requirements>
1. SSH to the server and create directory structure:
   - /home/ahs/grafana/
   - /home/ahs/grafana/instance/ (for persistent data)
   - /home/ahs/grafana/.gitignore with:
     ```
     instance/
     .env
     ```

2. Create docker-compose.yml with:
   - Grafana latest stable (grafana/grafana:latest)
   - Port 3000 exposed
   - Volume mount: ./instance:/var/lib/grafana
   - Environment variables for initial admin setup
   - Container name: grafana
   - Restart policy: unless-stopped
   - User mapping to avoid permission issues

3. Create .env file with:
   - GF_SECURITY_ADMIN_USER=admin
   - GF_SECURITY_ADMIN_PASSWORD=(generate random 24-char password)
   - GF_USERS_ALLOW_SIGN_UP=false

4. Ensure proper permissions on instance folder (Grafana runs as uid 472)

5. Start the stack and verify it's running:
   - docker compose up -d
   - docker compose ps (should show healthy)
   - Verify port 3000 is listening

6. Output the credentials clearly so the user can save them
</requirements>

<implementation>
Use SSH commands to create all files directly on the server.
Generate secure random strings using: openssl rand -base64 36 | tr -dc 'a-zA-Z0-9' | head -c N

The docker-compose.yml should:
- Use bind mount to ./instance
- Set proper user/group for Grafana (472:472)
- Include health check
- Don't hardcode secrets, use .env file
</implementation>

<output>
All files created on the remote server via SSH.
Display the generated credentials at the end for the user to save.
</output>

<verification>
Before declaring complete:
- Confirm container is running: docker compose ps shows "Up"
- Confirm port is accessible: curl http://localhost:3000/api/health returns JSON
- Display the admin credentials for user to record
</verification>

<success_criteria>
- Grafana container running on port 3000
- Data persisted to ./instance/ folder
- Credentials displayed for user to save
- Health check passes
</success_criteria>
