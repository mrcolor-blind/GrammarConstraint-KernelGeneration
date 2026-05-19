class MetricsParser:
    @staticmethod
    def parse_speedup(output: str):
        for line in output.splitlines():
            if line.startswith("speed up:"):
                try:
                    return float(
                        line.split(":", 1)[1].strip()
                    )
                except ValueError:
                    return None

        return None