from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Serving
    allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173,https://uno-bot.netlify.app"
    static_dir: str = ""
    allow_origins_regex: str = ""

    # Security
    api_key: str = ""
    trust_proxy: bool = False
    rate_start_per_min: int = 20
    rate_game_per_min: int = 90
    rate_replay_per_min: int = 60
    max_sessions: int = 400
    session_idle_ttl_min: int = 360
    session_finished_ttl_min: int = 30

    # RL bot
    training_dir: str = ""
    model_easy: str = ""
    model_normal: str = ""
    model_hard: str = ""
    bot_greedy: bool = True
    torch_threads: int = 4

    # Rules — mirror the training engine by default (see engine.py header)
    initial_hand_size: int = 10
    max_turns: int = 200

    # Replay persistence
    data_dir: str = ""
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = ""

    model_config = {"env_prefix": "UNO_", "env_file": ".env", "extra": "ignore"}

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allow_origins.split(",") if o.strip()]

    @property
    def r2_enabled(self) -> bool:
        return bool(self.r2_account_id and self.r2_access_key_id
                    and self.r2_secret_access_key and self.r2_bucket)


settings = Settings()