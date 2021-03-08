from matplotlib.backends.backend_agg import FigureCanvasAgg
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO

import matplotlib.figure as mplfigure
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import matplotlib as mpl
import requests
import numpy as np
import torch
import cv2
import os



class VisImage:

    def __init__(self, img, scale=1.0):
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)


    def _setup_figure(self, img):
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        # Need to imshow this first so that other patches can be drawn on top
        ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")
        self.fig = fig
        self.ax = ax


    def save(self, filepath):
        self.fig.savefig(filepath)


    def get_image(self):
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        buffer = np.frombuffer(s, dtype="uint8")
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


class Visualizer:

    @staticmethod
    def is_url(url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False


    @staticmethod
    def download_image(url):
        try:
            r = requests.get(url, stream=True)
        except Exception:
            return None, 400
        image_bytes = r.content
        return image_bytes, 200


    def read_image(self, f):
        if self.is_url(f):
            image_bytes, status_code = self.download_image(f)
            if status_code != 200:
                raise "Could not download image."
            f = BytesIO(image_bytes)

        image = Image.open(f)
        image = image.convert('RGB')
        return image


    def _set_canvas(self, f, scale=1.0):
        if isinstance(f, str):
            f = self.read_image(f)

        self.img = np.asarray(f).clip(0, 255).astype(np.uint8)
        self.output = VisImage(self.img, scale=scale)

        # too small texts are useless, therefore clamp to 9
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )


    @staticmethod
    def get_colors(num_instances, name='rainbow'):
        colors = plt.cm.get_cmap(name, num_instances)
        colors = [colors(i)[:3] for i in range(num_instances)]
        return colors


    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        x0, y0, width, height = box_coord
        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output


    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        if isinstance(segment, list):
            segment = np.asarray(segment).squeeze()

        if edge_color is None:
            edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
        )
        self.output.ax.add_patch(polygon)
        return self.output


    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0
    ):

        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark,
        # we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output


    def draw_label(self, label, score, bbox, color):
        if bbox is not None:
            x0, y0, w, h = bbox
            x1 = x0 + w
            y1 = y0 + h
            text_pos = (x0, y0)
            horiz_align = "left"
        else:
            return

        if score not in [None, [], -1]:
            label = "{} {:.0f}%".format(label, score * 100)

        # for small objects, draw text at the side to avoid occlusion
        instance_area = (y1 - y0) * (x1 - x0)
        if (
            instance_area < 1000 * self.output.scale
            or y1 - y0 < 40 * self.output.scale
        ):
            if y1 >= self.output.height - 5:
                text_pos = (x1, y0)
            else:
                text_pos = (x0, y1)

        height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
        font_size = (
            np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
            * 0.5
            * self._default_font_size
        )

        self.draw_text(
            label,
            text_pos,
            color=color,
            horizontal_alignment=horiz_align,
            font_size=font_size,
        )


    @staticmethod
    def export_detectrions(instance):
        bbox = instance['bbox']
        polygon = instance['segmentation']
        score = instance['score']
        category_id = instance['category_id']
        label = instance['label']
        return bbox, polygon, score, label


    def draw_detections(self, image, detections, save_as=None):
        self._set_canvas(image)
        assigned_colors = self.get_colors(len(detections))

        for i,instance in enumerate(detections):
            bbox, polygon, score, label = self.export_detectrions(instance)
            color = assigned_colors[i]

            if bbox not in [None, []]:
                self.draw_box(bbox, edge_color=color)
            if polygon not in [None, [], [[]]]:
                self.draw_polygon(polygon, color)
            if label not in [None, [], 'n/a']:
                self.draw_label(label, score, bbox, color)

        vis_image = self.output.get_image()[:, :, ::-1]

        if save_as:
            os.makedirs(os.path.dirname(save_as), exist_ok=True)
            cv2.imwrite(save_as, vis_image)

        return vis_image


    def __call__(self, image, detections, save_as=None):
        return self.draw_detections(image, detections, save_as)
