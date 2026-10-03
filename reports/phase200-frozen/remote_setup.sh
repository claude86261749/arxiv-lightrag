#!/bin/bash
set -e
sudo dnf install -y -q python3.11 python3.11-pip python3.11-devel postgresql16-server postgresql16-server-devel \
  postgresql16-contrib gcc make git redhat-rpm-config >/dev/null
echo PKGS_OK
cd /tmp && rm -rf pgvector && git clone -q --branch v0.8.0 --depth 1 https://github.com/pgvector/pgvector.git
cd pgvector && make -s >/dev/null && sudo make -s install >/dev/null && echo PGVECTOR_OK
if [ ! -f /var/lib/pgsql/data/PG_VERSION ]; then sudo postgresql-setup --initdb >/dev/null; fi
echo INITDB_OK
HBA=/var/lib/pgsql/data/pg_hba.conf
sudo sed -i -E 's#^(host\s+all\s+all\s+127\.0\.0\.1/32\s+)ident#\1scram-sha-256#; s#^(host\s+all\s+all\s+::1/128\s+)ident#\1scram-sha-256#' $HBA
CONF=/var/lib/pgsql/data/postgresql.conf
grep -q '^shared_buffers = 2GB' $CONF || printf 'shared_buffers = 2GB\nmax_connections = 200\n' | sudo tee -a $CONF >/dev/null
sudo systemctl enable --now postgresql >/dev/null 2>&1; sudo systemctl restart postgresql
echo PG_STARTED
sudo -u postgres psql -qc "ALTER USER postgres PASSWORD 'kgpipe'"
sudo -u postgres psql -Atc "SELECT 1 FROM pg_database WHERE datname='lightrag'" | grep -q 1 || sudo -u postgres psql -qc "CREATE DATABASE lightrag"
sudo -u postgres psql -d lightrag -qc "CREATE EXTENSION IF NOT EXISTS vector"
sudo -u postgres psql -d lightrag -Atc "select 'pgvector ' || extversion from pg_extension where extname='vector'"
grep -E '^host' $HBA
PGPASSWORD=kgpipe psql -h localhost -U postgres -d lightrag -Atc "select 'login ok'"
