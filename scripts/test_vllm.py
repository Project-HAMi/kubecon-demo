#!/usr/bin/env python3
"""
Test script for vLLM deployment
Connects to the vLLM service and sends a request
"""

import openai
import sys
import subprocess
import argparse
import signal
import time
import socket
import random
import logging

logger = logging.getLogger(__name__)

# Configuration
API_URL_LOCAL = "http://localhost:8000/v1"


def wait_for_port(port, host="localhost", timeout=10):
    """Wait for a port to become available"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return True
        except socket.error:
            pass
        time.sleep(0.5)
    return False


def start_port_forward(app_label, port):
    """Start kubectl port-forward and return process object"""
    logger.debug(f"Starting port-forward for {app_label} on port {port}")

    # Check if port is already in use
    if wait_for_port(port, timeout=1):
        logger.debug(f"⚠ Port {port} is already in use, attempting to use it anyway")

    # Start port-forward process
    cmd = ["kubectl", "port-forward", f"svc/{app_label}", f"{port}:vllm-svc"]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Wait for port-forward to establish with timeout
    logger.debug("Waiting for port-forward to establish (timeout: 10s)...")
    if wait_for_port(port, timeout=10):
        logger.debug(f"✓ Port-forward established on port {port}")
        return process
    else:
        # Port-forward failed to establish
        logger.debug(f"✗ Port-forward failed to establish on port {port}")
        try:
            process.terminate()
            process.wait(timeout=2)
        except:
            process.kill()
        raise ConnectionError(f"Port-forward failed to establish on port {port}")


def check_port_forward_process(process):
    """Check if port-forward process is still running"""
    if process is None:
        return False
    return_code = process.poll()
    if return_code is None:
        return True  # Process is still running
    else:
        # Process has terminated
        stdout, stderr = process.communicate()
        logger.debug(f"Port-forward process terminated with code {return_code}")
        if stderr:
            logger.debug(f"Stderr: {stderr[:200]}")
        return False


def restart_port_forward(app_label, port, old_process=None):
    """Restart port-forward process if needed"""
    if old_process:
        try:
            old_process.terminate()
            old_process.wait(timeout=2)
        except:
            try:
                old_process.kill()
            except:
                pass

    logger.debug(f"🔄 Restarting port-forward for {app_label} on port {port}")
    return start_port_forward(app_label, port)


def cleanup_port_forward(process):
    """Clean up port-forward process"""
    if process is None:
        return

    try:
        # Check if process is still running
        if process.poll() is None:
            # Try graceful termination first
            process.terminate()
            try:
                process.wait(timeout=3)
                logger.debug("✓ Port-forward cleaned up gracefully")
                return
            except subprocess.TimeoutExpired:
                # Force kill if not responding
                process.kill()
                process.wait(timeout=2)
                logger.debug("✓ Port-forward force killed")
        else:
            # Process already terminated
            stdout, stderr = process.communicate()
            if stderr:
                logger.debug(f"Port-forward stderr: {stderr[:100]}")
    except:
        pass  # Process already terminated or other error


# Global variable to track port-forward process for signal handling
_current_port_forward_process = None


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    logger.debug(f"\nReceived signal {signum}, cleaning up...")
    if _current_port_forward_process:
        cleanup_port_forward(_current_port_forward_process)
    sys.exit(1)


def test_api_connection(port, timeout=5):
    """Test basic TCP connection to API port"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except socket.error:
        return False


def test_vllm_api(app_label, model, port=8000):
    """Test vLLM API with a simple chat completion request"""
    global _current_port_forward_process
    port_forward_process = None
    max_port_forward_restarts = 30
    port_forward_restarts = 0

    try:
        # Set up signal handling
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        # Start initial port-forward
        port_forward_process = start_port_forward(app_label, port)
        _current_port_forward_process = port_forward_process
        API_URL_SERVICE = f"http://localhost:{port}/v1"

        # Try service URL first
        client = openai.OpenAI(
            base_url=API_URL_SERVICE,
            api_key="dummy",
        )
        logger.debug(f"Testing vLLM deployment with model: {model}")
        logger.debug(f"API URL (Service): {API_URL_SERVICE}")
        logger.debug("-" * 60)

        max_retries = 30  # 5 minutes total (30 * 10 seconds)
        retry_count = 0
        last_retry_time = time.time()

        while retry_count < max_retries:
            # Check if port-forward process is still alive
            if not check_port_forward_process(port_forward_process):
                if port_forward_restarts < max_port_forward_restarts:
                    port_forward_restarts += 1
                    logger.debug(
                        f"⚠ Port-forward died, restarting ({port_forward_restarts}/{max_port_forward_restarts})..."
                    )
                    port_forward_process = restart_port_forward(
                        app_label, port, port_forward_process
                    )
                    _current_port_forward_process = port_forward_process
                    # Reset retry count on successful restart
                    retry_count = 0
                    last_retry_time = time.time()
                else:
                    logger.debug(
                        f"✗ Max port-forward restarts ({max_port_forward_restarts}) exceeded"
                    )
                    raise ConnectionError("Port-forward keeps dying")

            # Test basic TCP connection first
            if not test_api_connection(port):
                logger.debug(f"⚠ TCP connection to port {port} failed")
                retry_count += 1
                elapsed_time = int(time.time() - last_retry_time)

                if retry_count == 1:
                    logger.debug("🔄 Waiting for vLLM to start up...")

                logger.debug(
                    f"   Retry {retry_count}/{max_retries} - Elapsed: {elapsed_time}s"
                )

                if retry_count < max_retries:
                    # Exponential backoff with jitter: 2s, 4s, 8s, 16s, then 10s
                    if retry_count <= 4:
                        sleep_time = 2**retry_count
                    else:
                        sleep_time = 10
                    # Add jitter (±20%)
                    jitter = sleep_time * 0.2
                    sleep_time = max(
                        1, sleep_time + (random.random() * 2 * jitter - jitter)
                    )
                    time.sleep(sleep_time)
                continue

            try:
                # Now test API connection
                test_response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1,
                )
                logger.debug("✓ Service URL accessible")
                break

            except Exception as service_error:
                retry_count += 1
                elapsed_time = int(time.time() - last_retry_time)

                if retry_count == 1:
                    logger.debug(f"⚠ Service URL not accessible: {service_error}")
                    logger.debug("🔄 Waiting for vLLM to start up...")

                logger.debug(
                    f"   Retry {retry_count}/{max_retries} - Elapsed: {elapsed_time}s"
                )

                if retry_count < max_retries:
                    # Exponential backoff with jitter
                    if retry_count <= 4:
                        sleep_time = 2**retry_count
                    else:
                        sleep_time = 10
                    jitter = sleep_time * 0.2
                    sleep_time = max(
                        1, sleep_time + (random.random() * 2 * jitter - jitter)
                    )
                    time.sleep(sleep_time)

        # Check if we exhausted retries
        if retry_count >= max_retries:
            logger.debug(f"✗ Max retries ({max_retries}) exceeded")
            logger.debug("\nPossible reasons:")
            logger.debug("1. vLLM deployment is not running")
            logger.debug("2. Service is not accessible")
            logger.debug("3. Port 8000 is not exposed")
            logger.debug("\nTo check deployment status:")
            logger.debug(f"  kubectl get pods -l app={app_label}")
            logger.debug(f"  kubectl logs -l app={app_label}")
            return 1

        # Main API test
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Say 'Hello from HAMi!' and briefly explain what vLLM is.",
                },
            ],
            max_tokens=100,
            temperature=0.7,
            stream=True,
        )

        logger.info("Response from vLLM:")
        logger.info("-" * 60)

        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)

        logger.info("\n" + "-" * 60)
        logger.info("✓ Test successful!")
        return 0

    except openai.OpenAIError as e:
        logger.error(f"OpenAI API Error: {e}")
        logger.debug("\nPossible reasons:")
        logger.debug("1. vLLM deployment is not running")
        logger.debug("2. Service is not accessible")
        logger.debug("3. Port 8000 is not exposed")
        logger.debug("\nTo check deployment status:")
        logger.debug(f"  kubectl get pods -l app={app_label}")
        logger.debug(f"  kubectl logs -l app={app_label}")
        return 1

    except ConnectionError as e:
        logger.error(f"Connection Error: {e}")
        logger.debug("\nTroubleshooting:")
        logger.debug("  1. Check if vLLM is running:")
        logger.debug(f"     kubectl get pods -l app={app_label}")
        logger.debug("  2. Check service:")
        logger.debug(f"     kubectl get svc {app_label}")
        logger.debug("  3. Try port-forward manually:")
        logger.debug(f"     kubectl port-forward svc/{app_label} {port}:8000")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

    finally:
        # Always clean up port-forward process
        try:
            cleanup_port_forward(port_forward_process)
            _current_port_forward_process = None
        except:
            pass


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="Test vLLM API with specified model")
    parser.add_argument(
        "--app",
        type=str,
        default="qwen8b",
        help="App label for the vLLM deployment (default: qwen8b)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name to test (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Local port for port-forward (default: 8000)",
    )
    args = parser.parse_args()

    sys.exit(test_vllm_api(args.app, args.model, args.port))
