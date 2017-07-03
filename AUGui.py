import copy
import glob
import os
import sys
from collections import defaultdict

import wx
from wx.lib.floatcanvas import NavCanvas

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts import AUScorer, OpenFaceScorer


class AUGui(wx.Frame):
    def __init__(self, parent, ID, name, directory, include_eyebrows=False):
        self.prominent_images = None
        self.skipping_index = 1
        wx.Frame.__init__(self, parent, ID, name)
        os.chdir(directory)
        self.images, self.imageIndex = self.make_images()
        self.original_images = copy.deepcopy(self.images)
        self.AU_threshold = 0
        self.scorer = AUScorer.AUScorer(directory, self.AU_threshold, include_eyebrows)

        n_c = NavCanvas.NavCanvas(self, Debug=0, BackgroundColor="BLACK")
        self.Canvas = n_c.Canvas

        self.curr_emotions = []
        self.emotional_pictures = [self.images[i] for i in self.scorer.emotions.keys()]
        self.AU_choices = copy.deepcopy(self.emotional_pictures)
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
        orderByProButton = wx.Button(self, wx.NewId(), label='Order By Prominence')
        show_landmarksButton = wx.Button(self, wx.NewId(), label='Show/Hide Landmarks')
        self.au_text = wx.StaticText(self, wx.NewId(), label='N/A')

        botBox.Add(orderByIndexButton, 1, wx.EXPAND)
        botBox.Add(orderByProButton, 1, wx.EXPAND)
        botBox.Add(show_landmarksButton, 1, wx.EXPAND)
        botBox.Add(self.au_text, 4, wx.EXPAND)

        self.allBox = wx.BoxSizer(wx.VERTICAL)
        self.allBox.Add(box, 3, wx.EXPAND)
        self.allBox.Add(botBox, 1, wx.EXPAND)

        # -- Make Bindings --
        self.Bind(wx.EVT_LISTBOX, self.click_on_pic, id=self.pic_box.GetId())
        self.Bind(wx.EVT_LISTBOX, self.click_on_emotion, id=self.AU_box.GetId())
        self.Bind(wx.EVT_BUTTON, self.order_by_index, id=orderByIndexButton.GetId())
        self.Bind(wx.EVT_BUTTON, self.order_by_pro, id=orderByProButton.GetId())
        self.Bind(wx.EVT_BUTTON, self.show_landmarks, id=show_landmarksButton.GetId())

        self.SetSizer(self.allBox)
        self.Layout()
        self.bind_to_canvas()

        # Landmark stuff
        self.landmarks_exist = False
        self.landmarks_shown = False
        marked_pics_dir = os.path.join(directory, 'labeled_frames/')
        if os.path.exists(marked_pics_dir):
            self.landmark_images = OpenFaceScorer.OpenFaceScorer.find_im_files(marked_pics_dir)
            if self.landmark_images:
                self.landmarks_exist = True

        self.update_all()

    def bind_to_canvas(self):
        self.Canvas.Bind(wx.EVT_KEY_DOWN, self.on_key_press)

    def click_on_emotion(self, event):
        emote_template = self.scorer.emotion_templates()
        label = ''
        for emote in sorted(emote_template.keys()):
            label += '{0} : {1} \n'.format(emote, str(emote_template[emote]))
        self.pop_dialog('Emotion Templates', label)
        self.AU_box.Deselect(self.AU_box.GetSelection())

    @staticmethod
    def make_images():
        images = sorted(glob.glob('*.png'))
        imageIndex = 0
        return images, imageIndex

    def click_on_pic(self, event):
        self.imageIndex = self.images.index(self.AU_choices[event.GetInt()])
        self.update_all()

    def show_landmarks(self, event):
        if self.landmarks_exist:
            self.landmarks_shown = not self.landmarks_shown
            self.show_im()
        else:
            self.pop_dialog('Landmark Error', 'No Landmarks Found!')

    def pop_dialog(self, name, string):
        dialog = wx.Dialog(self, wx.NewId(), name)
        textSizer = dialog.CreateTextSizer(string)
        dialog.SetSizer(textSizer)
        dialog.Show(True)

    def show_im(self):
        if self.landmarks_shown:
            image = self.landmark_images[self.imageIndex]
        else:
            image = self.images[self.imageIndex]
        self.Canvas.InitAll()
        curr_im = wx.Image(image)
        bm = curr_im.ConvertToBitmap()
        self.Canvas.AddScaledBitmap(bm, XY=(0, 0), Height=500, Position='tl')
        self.redraw()

    def order_by_index(self, event):
        self.AU_choices = self.emotional_pictures
        self.AU_box.Set(self.AU_choices)

    def order_by_pro(self, event):
        if not self.prominent_images:
            new_emotional_pics = []
            reverse_dict = {
                i: [] for i in range(6)
            }
            for frame, frame_dict in self.scorer.emotions.items():
                reverse_dict[max(frame_dict.values())].append(frame)
            for max_value in sorted(reverse_dict.keys(), reverse=True):
                for pic_num in reverse_dict[max_value]:
                    new_emotional_pics.append(self.images[pic_num])
            self.prominent_images = new_emotional_pics
        self.AU_choices = self.prominent_images
        self.pic_box.Set(self.AU_choices)

    def rewrite_text(self):
        label = 'CurrIm = ' + self.images[self.imageIndex] + '\n\n' + 'AUs' + '\n'
        presences = self.scorer.presence_dict
        name_dict = self.au_name_dict()
        if presences[self.imageIndex]:
            au_dict = presences[self.imageIndex]
            au_dict_keys = sorted(au_dict.keys())
            for au in au_dict_keys:
                if 'c' in au:
                    r_label = au.replace('c', 'r')
                    if r_label in au_dict_keys:
                        val = str(au_dict[r_label])
                    else:
                        val = 'Present'
                    au_int = AUScorer.AUScorer.return_num(au)
                    label += '{0} ({1}) = {2} \n'.format(str(au_int), name_dict[au_int], val)

        self.au_text.SetLabel(label)

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

    def au_name_dict(self):
        return {
            1: 'Inner Brow Raiser',
            2: 'Outer Brow Raiser',
            4: 'Brow Lowerer',
            5: 'Upper Lid Raiser',
            6: 'Cheek Raiser',
            7: 'Lid Tightener',
            9: 'Nose Wrinkler',
            10: 'Upper Lip Raiser',
            12: 'Lip Corner Puller',
            14: 'Dimpler',
            15: 'Lip Corner Depressor',
            17: 'Chin Raiser',
            20: 'Lip Stretcher',
            23: 'Lip Tightener',
            25: 'Lips Part',
            26: 'Jaw Drop',
            28: 'Lip Suck',
            45: 'Blink'
        }

    def update_au_list(self):
        if self.imageIndex in self.reverse_emotions.keys():
            total_arr = ['Scores']
            for value, emotions in sorted(self.reverse_emotions[self.imageIndex].items(), reverse=True):
                for emotion in emotions:
                    total_arr.append(emotion + ' = ' + str(value))
            self.AU_box.Set(total_arr)
        else:
            self.AU_box.Set(['None'])


if __name__ == '__main__':
    include_eyebrows = False
    if '-eb' in sys.argv:
        include_eyebrows = str(sys.argv[sys.argv.index('eb') + 1]) == '1'
    directory = sys.argv[sys.argv.index('-d') + 1]
    app = wx.App(False)
    score = AUGui(None, wx.ID_ANY, "AuGUI", directory)
    score.Show(True)
    app.MainLoop()
