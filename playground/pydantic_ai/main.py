from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the config-driven pydantic-ai FastAPI demo")
    parser.add_argument("--env-file", default="playground/pydantic_ai/.env")
    parser.add_argument("--yaml-file", default="playground/pydantic_ai/config/config.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    os.environ["PYDANTIC_AI_DEMO_ENV_FILE"] = args.env_file
    os.environ["PYDANTIC_AI_DEMO_YAML_FILE"] = args.yaml_file

    uvicorn.run(
        "playground.pydantic_ai.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
