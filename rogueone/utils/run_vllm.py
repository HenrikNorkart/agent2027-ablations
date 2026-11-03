import shlex
from pathlib import Path
import subprocess
import os
import signal
from typing import Optional
from time import sleep

from openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file

_EMBEDDING_SERVER_PID = None  # Stores the process ID of the embedding server


def start_daemon(
    cmd: str,
    pidfile: Optional[str] = None,
    logfile: Optional[str] = None,
    cwd: Optional[str] = None,
) -> int:
    """Start a command as a detached daemon, independent of terminal.
    
    Runs the command in a new process group so it survives terminal closure.
    Optionally writes the PID to a file and redirects output to a log file.
    
    Args:
        cmd (str): Shell command to execute.
        pidfile (str, optional): Path where to write the process ID.
        logfile (str, optional): Path where to write stdout/stderr. If None, discards output.
        cwd (str, optional): Working directory for the process.
        
    Returns:
        int: The process ID of the started daemon.
    """
    # Open logfile or /dev/null for child stdout/stderr
    if logfile:
        out = open(logfile, "ab")
        err = out
    else:
        out = open(os.devnull, "wb")
        err = out

    # Start the process detached from the controlling terminal
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=out,
        stderr=err,
        cwd=cwd,
        preexec_fn=os.setsid,
        close_fds=True,
    )

    pid = proc.pid

    # Write pidfile if requested (best-effort)
    if pidfile:
        try:
            Path(pidfile).write_text(str(pid))
        except Exception:
            # Intentionally ignore filesystem errors here; caller can handle if needed
            pass

    return pid


def run_script_daemon(
    script_path: str,
    cwd: str | None = None,
    pidfile: str | None = None,
    logfile: str | None = None,
) -> int:
    """Run a bash script as a detached daemon.
    
    Ensures the script exists and is executable before running. If cwd is not provided,
    uses the script's parent directory. Quotes the path safely for shell execution.
    
    Args:
        script_path (str): Path to the bash script to run.
        cwd (str, optional): Working directory. Defaults to script's parent directory.
        pidfile (str, optional): Path where to write the process ID.
        logfile (str, optional): Path where to write stdout/stderr output.
        
    Returns:
        int: The process ID of the started daemon.
        
    Raises:
        FileNotFoundError: If the script does not exist.
    """
    script = Path(script_path)
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    script = script.resolve()

    # Make executable if not already
    mode = script.stat().st_mode
    if not (mode & 0o111):
        script.chmod(mode | 0o111)

    # Use the script's directory as cwd by default
    if cwd is None:
        cwd = str(script.parent)

    # Quote the script path to be safe when using shell=True in start_daemon
    cmd = f"bash {shlex.quote(str(script))}"

    return start_daemon(cmd, pidfile=pidfile, logfile=logfile, cwd=cwd)


def start_vllm_servers(start_llm_server: bool = True, start_embedding_server: bool = True) -> int:
    """Start LLM and embedding vLLM servers as daemons.
    
    Launches both the main LLM server and embedding server using configured scripts.
    Waits for both servers to become responsive via test requests before returning.
    
    Args:
        start_llm_server (bool): Whether to start the LLM server.
        start_embedding_server (bool): Whether to start the embedding server.
    
    Returns:
        int: Status code (0 on success).
        
    Raises:
        FileNotFoundError: If server startup scripts are not found.
    """
    global _EMBEDDING_SERVER_PID
    
    assert start_llm_server or start_embedding_server, "At least one server must be started."

    LLM_server_script = Path("/workspace/scripts/run_vllm_oss120.sh")
    EMBEDD_server_script = Path("/workspace/scripts/run_vllm_qwen3_embedd.sh")

    if start_llm_server:
        assert (
            LLM_server_script.exists()
        ), f"LLM server script not found: {LLM_server_script}"
    if start_embedding_server:
        assert (
            EMBEDD_server_script.exists()
        ), f"Embedding server script not found: {EMBEDD_server_script}"

    if start_llm_server:
        run_script_daemon(
            script_path=str(LLM_server_script),
        )

    if start_embedding_server:
        _EMBEDDING_SERVER_PID = run_script_daemon(
            script_path=str(EMBEDD_server_script),
        )

    if start_llm_server:
        
        print("#### Waiting for VLLM servers to start... ####")
        
        base_url = f"http://localhost:{os.getenv('LLM_PORT')}/v1"
        api_key = os.getenv("LLM_API_KEY")
        model_name = os.getenv("LLM_MODEL")
        
        while True:
            try:
                client = OpenAI(
                    base_url=base_url,
                    api_key=api_key,
                )
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": "test"}],
                    model=model_name,
                )

                if response and isinstance(response.choices[0].message.content, str):
                    print(f"#### {model_name} servers are up and running! ####")
                    break
                
            except Exception as e:
                print(f"Waiting for {model_name} servers to be ready... {e}")
                print("Retrying in 10 seconds...")

            sleep(10)


    if start_embedding_server:
        print("#### Waiting for Embedding server to start... ####")

        while True:
            base_url = f"http://localhost:{os.getenv('EMBEDD_PORT')}/v1"
            api_key = os.getenv("EMBEDD_API_KEY")
            model_name = os.getenv("EMBEDD_MODEL")

            try:
                client = OpenAIEmbeddings(
                    model=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    tiktoken_enabled=True,
                )
                response = client.embed_documents(["test embedding"])

                if response and isinstance(response[0], list):
                    print(f"#### {model_name} server is up and running! ####")
                    break
            except Exception as e:
                print(f"Waiting for {model_name} server to be ready... {e}")
                print("Retrying in 10 seconds...")
            sleep(10)

    print()
    print("#######################################")
    print("#### All VLLM servers are running! ####")
    print("#######################################")
    print()

    return 0


def shutdown_embedding_server():
    """Shutdown the embedding server process to free GPU resources.
    
    Sends SIGTERM to the embedding server process group. Should be called after
    embeddings are cached to free GPU memory for model inference.
    """
    global _EMBEDDING_SERVER_PID

    os.killpg(
        _EMBEDDING_SERVER_PID, signal.SIGTERM  # Process group ID  # Termination signal
    )
    _EMBEDDING_SERVER_PID = None
