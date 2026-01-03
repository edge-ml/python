from __future__ import annotations

from typing import Optional


class Labeling:
    def __init__(self):
        self._id: Optional[int] = None
        self.labels: Optional[list[Label]] = None
        self.name: Optional[str] = None

    def parse(self, data: dict) -> None:
        self._id = data["labelingId"]
        self.name = data["name"]
        self.labels = []
        for label in data["labels"]:
            tmp_label = Label()
            tmp_label.parse(label)
            self.labels.append(tmp_label)

    def __str__(self) -> str:
        return f"Labeling(_id={self._id}, labels={self.labels}, name={self.name})"

    def __repr__(self) -> str:
        return str(self)


class Label:
    def __init__(self):
        self._id: Optional[int] = None
        self.start: Optional[int] = None
        self.end: Optional[int] = None
        self.type: Optional[int] = None
        self.name: Optional[str] = None

    def parse(self, data: dict) -> None:
        self._id = data.get("_id", data.get("id"))
        self.start = data["start"]
        self.end = data["end"]
        self.type = data["type"]
        self.name = data["name"]

    def __str__(self) -> str:
        return f"Label(_id={self._id}, start={self.start}, end={self.end}, type={self.type}, name={self.name})"

    def __repr__(self) -> str:
        return str(self)
