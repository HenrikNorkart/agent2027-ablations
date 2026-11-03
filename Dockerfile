# FROM nvcr.io/nvidia/pytorch:25.03-py3
# FROM vllm/vllm-openai:latest
FROM vllm/vllm-openai:v0.11.0


# Set an argument for the frontend, making it clear this is for build-time
ARG DEBIAN_FRONTEND=noninteractive
# Set the timezone to a default value like UTC to prevent prompts
ENV TZ=Etc/UTC


WORKDIR /workspace
COPY . .


RUN chmod +x /workspace/docker-entrypoint.sh


# AGENT 2027: pin every dependency nestedcv9 had to its exact version (pip freeze of
# rogue_one:nestedcv9), so v1 and v2 images differ only in code. Unpinned, the
# agent2027-pre.1 build pulled openai 2.54 / pydantic 2.14, which broke the
# openai-agents 0.9.1 usage parsing (every LLM call failed). New v2-only deps float.
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install -c constraints/nestedcv9.txt -e .


# In your Dockerfile
# RUN apt-get update && apt-get install -y python3-tk

# ENTRYPOINT [ "/bin/bash", "lc" ]

ENTRYPOINT ["/workspace/docker-entrypoint.sh"]
CMD ["python3 -u /workspace/rogue_one_main.py || tail -f /dev/null"]
