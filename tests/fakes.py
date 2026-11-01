class RecordingDucker:
    def __init__(self):
        self.calls = []

    def acquire(self, pid):
        self.calls.append("acquire")

    def duck(self, pid):
        self.calls.append("duck")

    def release(self):
        self.calls.append("release")

    def shutdown(self):
        pass
