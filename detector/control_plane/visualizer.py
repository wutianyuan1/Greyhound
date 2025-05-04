from prometheus_client import CollectorRegistry, Gauge, Counter, push_to_gateway


class ValueLogger(object):
    def __init__(self, pushgateway_url='http://127.0.0.1:9091'):
        self.registry = CollectorRegistry()
        self.pushgateway_url = pushgateway_url

    def push_val(self, metric_name, label, metric_vals):
        gauge = Gauge(metric_name, metric_name, [label], registry=self.registry)
        for lb, val in metric_vals.items():
            gauge.labels(rank=lb).set(val)
        # Push metrics to Prometheus Pushgateway
        push_to_gateway(self.pushgateway_url, job='failslow_metrics', registry=self.registry)


class CountLogger(object):
    def __init__(self, metric_name, pushgateway_url='http://127.0.0.1:9091'):
        self.registry = CollectorRegistry()
        self.pushgateway_url = pushgateway_url
        self.push_counter = Counter(metric_name, 'Number of times metrics are pushed', registry=self.registry)

    def inc_count(self):
        self.push_counter.inc()
