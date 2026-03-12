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
import os

# Configuration
API_URL_LOCAL = "http://localhost:8000/v1"


def start_port_forward(app_label):
    """Start kubectl port-forward in background"""
    cmd = ["kubectl", "port-forward", f"svc/{app_label}", "8000:8000"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Give port-forward time to establish
    time.sleep(3)

    return process.pid


def cleanup_port_forward(pid):
    """Clean up port-forward process"""
    try:
        os.kill(pid, signal.SIGTERM)
        print("✓ Port-forward cleaned up")
    except:
        pass  # Process already terminated


def test_vllm_api(app_label, model):
    """Test vLLM API with a simple chat completion request"""
    PORT_FORWARD_PID = start_port_forward(app_label)

    try:
        # Try service URL first
        client = openai.OpenAI(
            base_url=API_URL_LOCAL,
            api_key="dummy",
        )
        print(f"Testing vLLM deployment with model: {model}")
        print("-" * 60)

        max_retries = 30  # 5 minutes total (30 * 10 seconds)
        retry_count = 0

        while retry_count < max_retries:
            try:
                test_response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1,
                )
                print("✓ Service URL accessible")
                break

            except Exception as service_error:
                retry_count += 1
                elapsed_time = retry_count * 10

                if retry_count == 1:
                    print(f"⚠ Service URL not accessible: {service_error}")
                    print("🔄 Waiting for vLLM to start up...")

                print(
                    f"   Retry {retry_count}/{max_retries} - Elapsed: {elapsed_time}s"
                )

                if retry_count < max_retries:
                    time.sleep(10)  # Wait 10 seconds before retrying

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

        print("Response from vLLM:")
        print("-" * 60)

        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)

        print("\n" + "-" * 60)
        print("✓ Test successful!")

        try:
            cleanup_port_forward(PORT_FORWARD_PID)
        except:
            pass

        return 0

    except openai.OpenAIError as e:
        print(f"OpenAI API Error: {e}", file=sys.stderr)
        print("\nPossible reasons:")
        print("1. vLLM deployment is not running")
        print("2. Service is not accessible")
        print("3. Port 8000 is not exposed")
        print("\nTo check deployment status:")
        print("  kubectl get pods -l app=qwen8b")
        print("  kubectl logs -l app=qwen8b")
        return 1

    except ConnectionError as e:
        print(f"Connection Error: {e}", file=sys.stderr)
        print("\nTroubleshooting:")
        print("  1. Check if vLLM is running:")
        print("     kubectl get pods -l app=qwen8b")
        print("  2. Check service:")
        print("     kubectl get svc qwen8b-service")
        print("  3. Try port-forward:")
        print("     kubectl port-forward svc/qwen8b-service 8000:8000")
        return 1

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(test_vllm_api())
