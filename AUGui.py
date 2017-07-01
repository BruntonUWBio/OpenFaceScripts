import glob
import os
import sys
from collections import defaultdict

import wx
from wx.lib.floatcanvas import NavCanvas

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts import AUScorer


class AUGui(wx.Frame):
    def __init__(self, parent, ID, name, directory):
        self.skipping_index = 1
        wx.Frame.__init__(self, parent, ID, name)
        os.chdir(directory)
        self.images, self.imageIndex = self.make_images()
        self.scorer = AUScorer.AUScorer(directory)

        n_c = NavCanvas.NavCanvas(self, Debug=0, BackgroundColor="BLACK")
        self.Canvas = n_c.Canvas

        self.curr_emotions = []
        self.emotional_pictures = [self.images[i] for i in self.scorer.emotions.keys()]
        self.reverse_emotions = defaultdict()
        for frame in self.scorer.emotions:
            self.reverse_emotions[frame] = {}
            for emotion, value in self.scorer.emotions[frame].items():
                if value not in self.reverse_emotions[frame].keys():
                    self.reverse_emotions[frame][value] = [emotion]
                else:
                    self.reverse_emotions[frame][value].append(emotion)
        self.AU_box = wx.ListBox(self, wx.NewId(), style=wx.LC_REPORT | wx.SUNKEN_BORDER, name='List of Emotions',
                                 choices=self.curr_emotions)
        self.pic_box = wx.ListBox(self, wx.NewId(), style=wx.LC_REPORT | wx.SUNKEN_BORDER, name='Pictures',
                                  choices=self.emotional_pictures)

        box = wx.BoxSizer(wx.HORIZONTAL)
        box.Add(self.pic_box, 1, wx.EXPAND)
        box.Add(n_c, 3, wx.EXPAND)
        box.Add(self.AU_box, 1, wx.EXPAND)

        botBox = wx.BoxSizer(wx.HORIZONTAL)
        orderByIndexButton = wx.Button(self, wx.NewId(), label='Order By Index')
        # orderByProButton = wx.Button(self, wx.NewId(), label='Order By Prominence')
        self.au_text = wx.StaticText(self, wx.NewId(), label='N/A')

        botBox.Add(orderByIndexButton, 1, wx.EXPAND)
        # botBox.Add(orderByProButton, 1, wx.EXPAND)
        botBox.Add(self.au_text, 3, wx.EXPAND)

        self.allBox = wx.BoxSizer(wx.VERTICAL)
        self.allBox.Add(box, 3, wx.EXPAND)
        self.allBox.Add(botBox, 1, wx.EXPAND)

        self.Bind(wx.EVT_LISTBOX, self.click_on_pic, id=self.pic_box.GetId())
        self.Bind(wx.EVT_BUTTON, self.order_by_index, id=orderByIndexButton.GetId())
        # self.Bind(wx.EVT_BUTTON, self.order_by_pro, id=orderByProButton.GetId())

        self.SetSizer(self.allBox)
        self.Layout()
        self.update_all()
        self.bind_to_canvas()

    def bind_to_canvas(self):
        self.Canvas.Bind(wx.EVT_KEY_DOWN, self.on_key_press)

    @staticmethod
    def make_images():
        images = sorted(glob.glob('*.png'))
        imageIndex = 0
        return images, imageIndex

    def click_on_pic(self, event):
        self.imageIndex = self.images.index(self.emotional_pictures[event.GetInt()])
        self.update_all()

    def show_im(self):
        self.Canvas.InitAll()
        curr_im = wx.Image(self.images[self.imageIndex])
        bm = curr_im.ConvertToBitmap()
        self.Canvas.AddScaledBitmap(bm, XY=(0, 0), Height=500, Position='tl')
        self.redraw()

    def order_by_index(self, event):
        curr_image = self.emotional_pictures[self.imageIndex]
        self.emotional_pictures = sorted(self.emotional_pictures)
        self.imageIndex = self.emotional_pictures.index(curr_image)
        self.AU_box.Set(self.emotional_pictures)

    def order_by_pro(self, event):
        new_emotional_pics = []
        curr_image = self.emotional_pictures[self.imageIndex]
        reverse_dict = {
            i: [] for i in range(6)
        }
        for frame, frame_dict in self.scorer.emotions.items():
            reverse_dict[max(frame_dict.values())].append(frame)
        for max_value in sorted(reverse_dict.keys(), reverse=True):
            for pic_num in reverse_dict[max_value]:
                try:
                    new_emotional_pics.append(self.emotional_pictures[pic_num])
                except IndexError:
                    print('what')
        self.emotional_pictures = new_emotional_pics
        self.imageIndex = self.emotional_pictures.index(curr_image)
        self.AU_box.Set(self.emotional_pictures)

    def rewrite_text(self):
        self.au_text.SetLabel('CurrIm = ' + self.images[self.imageIndex])

    def redraw(self):
        self.Canvas.Draw()
        self.Canvas.ZoomToBB()

    def on_key_press(self, event):
        keyCode = event.GetKeyCode()
        if keyCode == wx.WXK_RIGHT:
            while wx.GetKeyState(wx.WXK_RIGHT) and (self.imageIndex + self.skipping_index) in range(len(self.images)):
                self.imageIndex += self.skipping_index
                self.update_all()
        elif keyCode == wx.WXK_LEFT:
            while wx.GetKeyState(wx.WXK_LEFT) and (self.imageIndex - self.skipping_index) in range(len(self.images)):
                self.imageIndex -= self.skipping_index
                self.update_all()

    def update_all(self):
        self.show_im()
        self.update_au_list()
        self.rewrite_text()

    def update_au_list(self):
        if self.imageIndex in self.reverse_emotions.keys():
            total_arr = ['Scores']
            for value, emotions in sorted(self.reverse_emotions[self.imageIndex].items(), reverse=True):
                for emotion in emotions:
                    total_arr.append(emotion + ' = ' + str(value))
            # emotion_dict = self.scorer.emotions[self.imageIndex]
            # exact_emotions = emotion_dict['Exact Match']
            # possible_emotions = emotion_dict['Possible Match']
            # total_arr = ['Exact'] + exact_emotions + ['Possible'] + possible_emotions
            self.AU_box.Set(total_arr)
        else:
            self.AU_box.Set(['None'])


if __name__ == '__main__':
    directory = sys.argv[sys.argv.index('-d') + 1]
    app = wx.App(False)
    score = AUGui(None, wx.ID_ANY, "AuGUI", directory)
    score.Show(True)
    app.MainLoop()
