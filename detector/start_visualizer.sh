pushgateway &
prometheus --config.file=./prometheus.yml --web.listen-address=:9969 &
grafana-server --homepath /usr/share/grafana
