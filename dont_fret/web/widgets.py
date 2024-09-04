import base64

import ipyvuetify as v


class FloatField(v.TextField):
    def __init__(self, **kwargs):
        attributes = kwargs.pop("attributes", {"step": "any"})
        super().__init__(type="number", attributes=attributes, **kwargs)
        self.on_event("input", self.on_input)

    @property
    def value(self) -> float:
        return float(self.v_model)

    def on_input(self, widget, event, data):
        vmin, vmax = self.attributes.get("min"), self.attributes.get("max")
        self.error_messages = []
        try:
            value = self.value
            if vmin is not None and self.value < vmin:
                self.error_messages = [f"Input value must be larger than {vmin}"]
            elif vmax is not None and self.value > vmax:
                self.error_messages = [f"Input value must be smaller than {vmax}"]
        except ValueError:
            self.error_messages = ["Input value must be a floating point number"]


class IntField(v.TextField):
    def __init__(self, **kwargs):
        attributes = kwargs.pop("attributes", {"step": 1})

        super().__init__(type="number", attributes=attributes, **kwargs)
        self.on_event("input", self.on_input)

    @property
    def value(self) -> int:
        return int(self.v_model)

    def on_input(self, widget, event, data):
        vmin, vmax = self.attributes.get("min"), self.attributes.get("max")
        self.error_messages = []
        try:
            value = self.value
            if vmin is not None and self.value < vmin:
                self.error_messages = [f"Input value must be larger than {vmin}"]
            elif vmax is not None and self.value > vmax:
                self.error_messages = [f"Input value must be smaller than {vmax}"]
        except ValueError:
            self.error_messages = ["Input value must be an integer number"]


class ProgressLinearTasks(v.ProgressLinear):
    def __init__(self, num_tasks: int = 10, height=25, **kwargs):
        self.completed = 0
        self.num_tasks = num_tasks
        self.val_txt = v.Html(children=[self.bar_text], tag="strong")
        super().__init__(
            value=0.0,
            height=height,
            v_slots=[{"name": "default", "children": self.val_txt}],
            **kwargs,
        )

        self.observe(self.update_text, names="value")

    def update_text(self, *event):
        pass
        self.val_txt.children = [f"{self.completed}/{self.num_tasks}"]

    @property
    def bar_text(self) -> str:
        if self.num_tasks:
            return f"{self.completed}/{self.num_tasks}"
        else:
            return ""

    def reset(self):
        self.completed = 0
        self.num_tasks = 0
        self.value = 0

    def increment(self):
        if self.completed < self.num_tasks:
            self.completed += 1
            self.value = (self.completed / self.num_tasks) * 100


class FileDownloadBtn(v.Btn):
    """A button which can be given data to download on click.
    Usage:
        btn = FileDownloadBtn(children=['Download BUTTON'], attributes={'download': 'test.txt'})

    """

    # from: Holoviz panel's FileDownload.
    _mime_types = {
        "application": {"pdf": "pdf", "zip": "zip"},
        "audio": {"mp3": "mp3", "ogg": "ogg", "wav": "wav", "webm": "webm"},
        "image": {
            "apng": "apng",
            "bmp": "bmp",
            "gif": "gif",
            "ico": "x-icon",
            "cur": "x-icon",
            "jpg": "jpeg",
            "jpeg": "jpeg",
            "png": "png",
            "svg": "svg+xml",
            "tif": "tiff",
            "tiff": "tiff",
            "webp": "webp",
        },
        "text": {
            "css": "css",
            "csv": "plain;charset=UTF-8",
            "js": "javascript",
            "html": "html",
            "txt": "plain;charset=UTF-8",
        },
        "video": {"mp4": "mp4", "ogg": "ogg", "webm": "webm"},
    }

    def set_payload(self, byte_data, filename: str):
        b64 = base64.b64encode(byte_data).decode("utf-8")

        ext = filename.split(".")[1]
        for mtype, subtypes in self._mime_types.items():
            stype = None
            if ext in subtypes:
                stype = subtypes[ext]
                break
        if stype is None:
            mime = "application/octet-stream"
        else:
            mime = "{type}/{subtype}".format(type=mtype, subtype=stype)

        href = "data:{mime};base64,{b64}".format(mime=mime, b64=b64)

        self.attributes["download"] = filename
        self.href = href
