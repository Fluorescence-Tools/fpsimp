"""
Email notification service for FPSIMP.
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from config import config

def send_job_notification(job_id: str, event: str, message: str, redis_client):
    """Send email notification for job event"""
    job_data = redis_client.hgetall(f"job:{job_id}")
    email = job_data.get('email')
    
    if not email:
        return False
    
    try:
        # Build results URL
        base_url = config.HOST if config.HOST != '0.0.0.0' else 'localhost'
        if config.PORT:
            base_url = f"{base_url}:{config.PORT}"
        results_url = f"http://{base_url}/results/{job_id}"
        
        # Build email content
        subject = f"FPsimP Job {event}: {job_id[:8]}"
        
        body = f"""
Job ID: {job_id}
Status: {event}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{message}

View your results here: {results_url}

IMPORTANT: Job results are only retained for {config.CLEANUP_JOBS_AFTER_DAYS} days.

---
FPsimP - Fluorescent Protein Simulation Pipeline
"""
        
        # Send email
        if config.SMTP_SERVER and config.SMTP_ENABLED:
            send_email(email, subject, body)
        else:
            print(f"[EMAIL] To: {email}")
            print(f"[EMAIL] Subject: {subject}")
            print(f"[EMAIL] Body: {body}")
        
        return True
    except Exception as e:
        print(f"Failed to send email notification: {e}")
        return False

def send_email(to_addr: str, subject: str, body: str):
    """Send email via SMTP"""
    smtp_server = config.SMTP_SERVER
    smtp_port = config.SMTP_PORT
    smtp_user = config.SMTP_USER
    smtp_pass = config.SMTP_PASS
    from_addr = config.SMTP_FROM
    
    if not smtp_server:
        print(f"[EMAIL] Would send to {to_addr}: {subject}")
        return
    
    msg = MIMEMultipart()
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            if smtp_user and smtp_pass:
                server.starttls()
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"[EMAIL] Sent to {to_addr}")
    except Exception as e:
        print(f"[EMAIL] Failed to send: {e}")
