#!/bin/bash
set -e
cd ~/arxiv-lightrag && mkdir -p backups
T=$(date -u +%Y%m%dT%H%M%SZ)
PGPASSWORD=kgpipe pg_dump -h localhost -U postgres -Fc -f backups/lightrag-$T.dump lightrag
tar czf backups/phase200-$T.tar.gz runs/phase200 data/corpus
cd backups && sha256sum lightrag-$T.dump phase200-$T.tar.gz > SHA256SUMS && ls -la && cat SHA256SUMS
# verify the dump is readable
pg_restore -l lightrag-$T.dump | grep -c "TABLE DATA" | sed 's/^/tables in dump: /'
sudo systemctl stop postgresql && echo "postgres stopped: $(systemctl is-active postgresql)"
