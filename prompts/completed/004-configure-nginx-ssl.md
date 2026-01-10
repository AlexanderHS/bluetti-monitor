<objective>
Configure DNS records, SSL certificates, and Nginx reverse proxy for Grafana and InfluxDB.

This exposes the monitoring services over HTTPS at graf.ad.2ho.me and flux.ad.2ho.me.
</objective>

<context>
Production server: ssh ahs@blu (10.0.0.142)
DNS server (Samba AD/DC): ssh ahs@dc1
Nginx config: /etc/nginx/sites-enabled/default

Services to expose:
- Grafana: http://localhost:3000 → https://graf.ad.2ho.me
- InfluxDB: http://localhost:8086 → https://flux.ad.2ho.me

DNS zone: ad.2ho.me
DNS add command: sudo samba-tool dns add localhost ad.2ho.me <name> A <ip> -P
</context>

<prerequisites>
Before running this prompt:
- Prompts 001 and 002 must be complete (InfluxDB and Grafana running on blu)
</prerequisites>

<requirements>
1. Create DNS A records on dc1:
   ```bash
   ssh ahs@dc1 "sudo samba-tool dns add localhost ad.2ho.me graf A 10.0.0.142 -P"
   ssh ahs@dc1 "sudo samba-tool dns add localhost ad.2ho.me flux A 10.0.0.142 -P"
   ```

2. Verify DNS resolution:
   ```bash
   ssh ahs@dc1 "host graf.ad.2ho.me && host flux.ad.2ho.me"
   ```

3. Generate SSL certificates on blu (using Cloudflare DNS validation):
   ```bash
   ssh ahs@blu "sudo certbot certonly --dns-cloudflare --dns-cloudflare-credentials /etc/letsencrypt/cloudflare.ini -d graf.ad.2ho.me --email alexhamiltonsmith@gmail.com --agree-tos --non-interactive"
   ssh ahs@blu "sudo certbot certonly --dns-cloudflare --dns-cloudflare-credentials /etc/letsencrypt/cloudflare.ini -d flux.ad.2ho.me --email alexhamiltonsmith@gmail.com --agree-tos --non-interactive"
   ```

4. Add Nginx server blocks to /etc/nginx/sites-enabled/default on blu:
   - Follow the existing pattern (HTTP redirect + HTTPS with proxy)
   - graf.ad.2ho.me → localhost:3000 (Grafana)
   - flux.ad.2ho.me → localhost:8086 (InfluxDB)

5. Test and reload Nginx:
   ```bash
   ssh ahs@blu "sudo nginx -t && sudo systemctl reload nginx"
   ```

6. Verify HTTPS access:
   ```bash
   curl -I https://graf.ad.2ho.me
   curl -I https://flux.ad.2ho.me/health
   ```
</requirements>

<nginx_config>
Append these server blocks to /etc/nginx/sites-enabled/default:

```nginx
server {
    listen 80;
    server_name graf.ad.2ho.me;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name graf.ad.2ho.me;

    ssl_certificate /etc/letsencrypt/live/graf.ad.2ho.me/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/graf.ad.2ho.me/privkey.pem;

    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (required for Grafana Live)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Increase timeouts for long-running requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}

server {
    listen 80;
    server_name flux.ad.2ho.me;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name flux.ad.2ho.me;

    ssl_certificate /etc/letsencrypt/live/flux.ad.2ho.me/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/flux.ad.2ho.me/privkey.pem;

    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    location / {
        proxy_pass http://127.0.0.1:8086;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Increase timeouts for long-running requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```
</nginx_config>

<implementation>
Execute commands via SSH to both servers:
- dc1 for DNS record creation
- blu for SSL certs and Nginx config

For appending Nginx config, use heredoc with sudo tee -a:
```bash
ssh ahs@blu "sudo tee -a /etc/nginx/sites-enabled/default << 'EOF'
[nginx config here]
EOF"
```

Certbot may require interactive confirmation - handle appropriately.
</implementation>

<verification>
Before declaring complete:
1. DNS resolves: `host graf.ad.2ho.me` and `host flux.ad.2ho.me` return 10.0.0.142
2. Nginx config valid: `sudo nginx -t` returns success
3. HTTPS works: `curl -I https://graf.ad.2ho.me` returns 200 or 302
4. InfluxDB health: `curl https://flux.ad.2ho.me/health` returns JSON with status "pass"
</verification>

<success_criteria>
- DNS A records created for graf.ad.2ho.me and flux.ad.2ho.me
- SSL certificates generated via Let's Encrypt
- Nginx configured with HTTPS reverse proxy
- Both services accessible over HTTPS
</success_criteria>
