class TritonBenchAdapter:
    @staticmethod
    def normalize_summary(summary: dict):
        return {
            "total_predictions":
                summary["total_predictions"],

            "call_acc_passed":
                summary["call_acc"]["passed"],

            "call_acc_rate":
                summary["call_acc"]["rate"],

            "exec_acc_passed":
                summary["exec_acc"]["passed"],

            "exec_acc_rate":
                summary["exec_acc"]["rate"],

            "speedup":
                summary["speedup"],
        }