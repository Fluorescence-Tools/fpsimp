# Email Notification Setup

FPsimP can send email notifications when jobs are completed or fail. To enable this feature, you need to configure an external SMTP service.

## Configuration

Settings are managed via environment variables in the `.env` file.

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SMTP_ENABLED` | Enable or disable email notifications | `true` |
| `SMTP_SERVER` | SMTP server hostname | `smtp.sendgrid.net` |
| `SMTP_PORT` | SMTP server port (usually 587 for TLS) | `587` |
| `SMTP_USER` | SMTP username | `apikey` |
| `SMTP_PASS` | SMTP password or API key | `your_api_key_here` |
| `SMTP_FROM` | The "From" address for notifications | `noreply@yourdomain.com` |

## Service Examples

### SendGrid
- **SMTP Server**: `smtp.sendgrid.net`
- **Port**: `587`
- **Username**: `apikey`
- **Password**: Your SendGrid API Key

### Mailgun
- **SMTP Server**: `smtp.mailgun.org`
- **Port**: `587`
- **Username**: `postmaster@your.domain.com`
- **Password**: Your Mailgun SMTP password

### Gmail (App Password)
> [!WARNING]
> Using a personal Gmail account is not recommended for production. If you do, you MUST use an "App Password".
- **SMTP Server**: `smtp.gmail.com`
- **Port**: `587`
- **Username**: Your Gmail address
- **Password**: Your 16-character App Password

## Verification

After configuring the variables, restart the application:

```bash
docker-compose up -d
```

You can verify the setup by submitting a job with your email address. Check the application logs if emails are not being received:

```bash
docker-compose logs -f web-app
```

Look for `[EMAIL] Sent to ...` or `[EMAIL] Failed to send: ...` in the logs.

## Result Retention

> [!IMPORTANT]
> By default, job results are retained for **7 days**. This can be adjusted using the `CLEANUP_JOBS_AFTER_DAYS` variable in the `.env` file.
