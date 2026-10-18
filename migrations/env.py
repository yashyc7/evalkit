from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from evalkit.config import settings
from evalkit.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Use the database URL from our own settings (.env), not from alembic.ini.
# This way there is only ONE place where the URL lives.
config.set_main_option("sqlalchemy.url", settings.database_url)

# Alembic compares this metadata (our Python classes) with the real database
# to find out what is missing.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Offline mode: only prints the SQL, does not connect to the database."""
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Online mode: connects to the database and applies the changes."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
