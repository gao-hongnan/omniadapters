from __future__ import annotations

import argparse
import os

import uvicorn

from .constants import ENV_FILE_VAR, YAML_FILE_VAR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the config-driven pydantic-ai FastAPI demo")
    parser.add_argument("--env-file", default="playground/pydantic_ai/.env")
    parser.add_argument("--yaml-file", default="playground/pydantic_ai/config/config.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    os.environ[ENV_FILE_VAR] = args.env_file
    os.environ[YAML_FILE_VAR] = args.yaml_file

    uvicorn.run(
        "playground.pydantic_ai.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
