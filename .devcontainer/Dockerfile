# FROM nvcr.io/nvidia/pytorch:25.03-py3
# FROM vllm/vllm-openai:latest
FROM vllm/vllm-openai:v0.11.0


# Set an argument for the frontend, making it clear this is for build-time
ARG DEBIAN_FRONTEND=noninteractive
# Set the timezone to a default value like UTC to prevent prompts
ENV TZ=Etc/UTC


# In your Dockerfile
RUN apt-get update && apt-get install -y python3-tk

