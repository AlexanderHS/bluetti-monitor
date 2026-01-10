<objective>
Configure Nginx reverse proxy with SSL certificates for Grafana and InfluxDB on the production server.

This exposes the monitoring services over HTTPS using Let's Encrypt certificates via Cloudflare DNS validation.
</objective>

<context>
Server: ssh ahs@blu (10.0.0.142)
Nginx config: /etc/nginx/sites-enabled/default

Services to expose:
- Grafana: http://localhost:3000 → https://graf.ad.2ho.me
- InfluxDB: http://localhost:8086 → https://flux.ad.2ho.me

SSL cert generation command:
sudo certbot certonly --dns-cloudflare --dns-cloudflare-credentials /etc/letsencrypt/cloudflare.ini -d "hostname.ad.2ho.me"

IMPORTANT: The user will manually create DNS entries for graf.ad.2ho.me and flux.ad.2ho.me pointing to this server. Certbot with Cloudflare DNS validation will work regardless of DNS propagation for the A record.
</context>

<prerequisites>
Before running this prompt:
- Prompts 001 and 002 must be complete (InfluxDB and Grafana running)
- User must have created DNS entries for graf.ad.2ho.me and flux.ad.2ho.me
</prerequisites>

<requirements>
1. Generate SSL certificates for both hostnames:
   ```
   sudo certbot certonly --dns-cloudflare --dns-cloudflare-credentials /etc/letsencrypt/cloudflare.ini -d "graf.ad.2ho.me"
   sudo certbot certonly --dns-cloudflare --dns-cloudflare-credentials /etc/letsencrypt/cloudflare.ini -d "flux.ad.2ho.me"
   ```

2. Add Nginx server blocks to /etc/nginx/sites-enabled/default following the existing pattern:

   For Grafana (graf.ad.2ho.me → localhost:3000):
   - HTTP server block redirecting to HTTPS
   - HTTPS server block with:
     - SSL certs from /etc/letsencrypt/live/graf.ad.2ho.me/
     - proxy_pass to http://127.0.0.1:3000
     - WebSocket support (required for Grafana live features)
     - Standard proxy headers

   For InfluxDB (flux.ad.2ho.me → localhost:8086):
   - HTTP server block redirecting to HTTPS
   - HTTPS server block with:
     - SSL certs from /etc/letsencrypt/live/flux.ad.2ho.me/
     - proxy_pass to http://127.0.0.1:8086
     - Standard proxy headers

3. Test Nginx configuration:
   sudo nginx -t

4. Reload Nginx:
   sudo systemctl reload nginx

5. Verify both services are accessible:
   - curl -I https://graf.ad.2ho.me (should return 200 or 302 to login)
   - curl -I https://flux.ad.2ho.me/health (should return 200)
</requirements>

<nginx_template>
Follow this exact pattern from the existing config:

```nginx
server {
    listen 80;
    server_name HOSTNAME.ad.2ho.me;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name HOSTNAME.ad.2ho.me;

    ssl_certificate /etc/letsencrypt/live/HOSTNAME.ad.2ho.me/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/HOSTNAME.ad.2ho.me/privkey.pem;

    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    location / {
        proxy_pass http://127.0.0.1:PORT;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Optional: Increase timeouts for long-running requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```
</nginx_template>

<implementation>
Use SSH to execute all commands on the server.

For editing /etc/nginx/sites-enabled/default:
- Use sudo to append the new server blocks
- Or use a heredoc with sudo tee -a

The certbot command requires sudo and will prompt for confirmation - use appropriate flags or expect interactive prompts.
</implementation>

<output>
On the server:
- SSL certs generated for graf.ad.2ho.me and flux.ad.2ho.me
- Nginx config updated with new server blocks
- Nginx reloaded

Verify with curl commands showing HTTPS access works.
</output>

<verification>
Before declaring complete:
1. sudo nginx -t returns "syntax is ok" and "test is successful"
2. curl -I https://graf.ad.2ho.me returns HTTP response (200 or 302)
3. curl -I https://flux.ad.2ho.me/health returns 200 with JSON
4. Both services accessible in browser (user can verify manually)
</verification>

<success_criteria>
- SSL certificates generated for both hostnames
- Nginx configured and reloaded without errors
- Both services accessible over HTTPS
- WebSocket support enabled for Grafana
</success_criteria>
