"""
.. module:: AUGui
    :synopsis: A GUI for viewing a picture and the associated action units and emotion prediction.
"""

import csv
import glob
import os
import sys

import numpy as np
import wx
from wx.lib.floatcanvas import NavCanvas

sys.path.append('/home/gvelchuru/')
from OpenFaceScripts import AUScorer, OpenFaceScorer


class AUGui(wx.Frame):
    """
    Main GUI
    """
    def __init__(self, parent, frame_id, name, curr_directory, incl_eyebrows=False, path_to_csv=None):
        self.path_to_csv = path_to_csv
        self.prominent_images = None
        self.skipping_index = 1
        wx.Frame.__init__(self, parent, frame_id, name)
        os.chdir(curr_directory)
        self.images, self.imageIndex = make_images()

        self.image_map = None
        self.annotated_map = None
        self.all_shown = True
        if self.path_to_csv:
            self.image_map = csv_emotion_reader(path_to_csv)
            self.annotated_map = {self.images[index * 30]: emotion for index, emotion in
                                  enumerate(sorted(self.image_map.values())) if
                                  (index * 30) < len(self.images)}  # Relies on image map only having one item per image

        self.AU_threshold = 0
        self.scorer = AUScorer.AUScorer(curr_directory, self.AU_threshold, incl_eyebrows)

        n_c = NavCanvas.NavCanvas(self, Debug=0, BackgroundColor="BLACK")
        self.Canvas = n_c.Canvas

        self.curr_emotions = []
        self.AU_choices = self.make_full_au_choices()
        self.AU_box = wx.BoxSizer(wx.VERTICAL)
        self.AU_List = wx.ListBox(self, wx.NewId(), style=wx.LC_REPORT | wx.SUNKEN_BORDER, name='List of Emotions',
                                  choices=self.curr_emotions)
        self.AU_box.Add(self.AU_List, 3, wx.EXPAND)
        if self.path_to_csv:
            self.annotation_box = wx.TextCtrl(self, wx.NewId(), value='N/A', style=wx.TE_READONLY | wx.TE_MULTILINE)
            self.AU_box.Add(self.annotation_box, 1, wx.EXPAND)
        self.pic_box = wx.ListBox(self, wx.NewId(), style=wx.LC_REPORT | wx.SUNKEN_BORDER, name='Pictures',
                                  choices=self.AU_choices)

        box = wx.BoxSizer(wx.HORIZONTAL)
        box.Add(self.pic_box, 1, wx.EXPAND)
        box.Add(n_c, 3, wx.EXPAND)
        box.Add(self.AU_box, 1, wx.EXPAND)

        botBox = wx.BoxSizer(wx.HORIZONTAL)
        self.order = 'Index'
        self.order_button = wx.Button(self, wx.NewId(), label='Order By Prominence')
        show_landmarksButton = wx.Button(self, wx.NewId(), label='Show/Hide Landmarks')
        self.au_text = wx.TextCtrl(self, wx.NewId(), value='N/A', style=wx.VSCROLL | wx.TE_READONLY | wx.TE_MULTILINE)

        show_vidButton = wx.Button(self, wx.NewId(), label='Show Video Around Frame')

        botBox.Add(self.order_button, 1, wx.EXPAND)
        botBox.Add(show_landmarksButton, 1, wx.EXPAND)
        if self.path_to_csv:
            self.show_annotations_button = wx.Button(self, wx.NewId(), label='Show Annotated Frames')
            botBox.Add(self.show_annotations_button, 1, wx.EXPAND)
            self.Bind(wx.EVT_BUTTON, self.show_hide_annotations, id=self.show_annotations_button.GetId())
        botBox.Add(show_vidButton, 1, wx.EXPAND)
        botBox.Add(self.au_text, 4, wx.EXPAND)

        self.allBox = wx.BoxSizer(wx.VERTICAL)
        self.allBox.Add(box, 4, wx.EXPAND)
        self.allBox.Add(botBox, 1, wx.EXPAND)

        # -- Make Bindings --
        self.Bind(wx.EVT_LISTBOX, self.click_on_pic, id=self.pic_box.GetId())
        self.Bind(wx.EVT_LISTBOX, self.click_on_emotion, id=self.AU_List.GetId())
        self.Bind(wx.EVT_BUTTON, self.evt_reorder_pics, id=self.order_button.GetId())
        self.Bind(wx.EVT_BUTTON, self.show_landmarks, id=show_landmarksButton.GetId())
        self.Bind(wx.EVT_BUTTON, self.show_video, id=show_vidButton.GetId())

        self.SetSizer(self.allBox)
        self.Layout()
        self.bind_to_canvas()

        # Landmark stuff
        self.landmarks_exist = False
        self.landmarks_shown = False
        marked_pics_dir = os.path.join(curr_directory, 'labeled_frames/')
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
        self.AU_List.Deselect(self.AU_List.GetSelection())

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
        """Shows a new text dialog with name equal to name and text equal to string.

        :param name: Name of dialog.
        :param string: Text to display.
        :return: None
        """
        dialog = wx.Dialog(self, wx.NewId(), name)
        textSizer = dialog.CreateTextSizer(string)
        dialog.SetSizer(textSizer)
        dialog.Show(True)

    def show_im(self):
        """Changes currently displayed image to the image at self.imageIndex

        :return: None
        """
        if self.landmarks_shown:
            image = self.landmark_images[self.imageIndex]
        else:
            image = self.images[self.imageIndex]
        self.Canvas.InitAll()
        curr_im = wx.Image(image)
        bm = curr_im.ConvertToBitmap()
        self.Canvas.AddScaledBitmap(bm, XY=(0, 0), Height=500, Position='tl')
        self.redraw()

    def show_video(self, event):
        pass

    def evt_reorder_pics(self, event):
        """Event handling wrapper for reordering pictures. If pictures are currently ordered by index, orders by
        prominence and vice versa.

        :param event: Unused.
        :return: None
        """
        if self.order == 'Index':
            self.reorder_pics('Prominence')
        else:
            self.reorder_pics('Index')

    def reorder_pics(self, order_type):
        """Reorders pictures according to the specified order type.

        :param order_type: Either 'Index' or 'Prominence'.
        :return: None
        """
        if order_type == 'Index':
            self.AU_choices.sort()
            self.order_button.SetLabel('Order by Prominence')
        elif order_type == 'Prominence':
            self.AU_choices.sort(key=lambda pic: (
                self.prevalence_score(self.scorer.get_emotions(self.images.index(pic))) if self.images.index(
                    pic) in self.scorer.emotions.keys() else 0),
                                 reverse=True)
            self.order_button.SetLabel('Order by Index')
        else:
            raise ValueError('Unknown order type')
        self.order = order_type
        self.pic_box.Set(self.AU_choices)

    def rewrite_text(self):
        label = 'Current Image = ' + self.images[self.imageIndex] + '\n\n' + 'AUs' + '\n'
        name_dict = au_name_dict()
        if self.scorer.presence_dict[self.imageIndex]:
            au_dict = self.scorer.presence_dict[self.imageIndex]
            au_dict_keys = sorted(au_dict.keys())
            for au in au_dict_keys:
                if 'c' in au:
                    r_label = au.replace('c', 'r')
                    au_int = AUScorer.AUScorer.return_num(au)
                    label += '{0} ({1}) = {2} \n'.format(str(au_int), name_dict[au_int], str(
                        au_dict[r_label]) if r_label in au_dict_keys else 'Present')
        self.au_text.SetValue(label)

    def redraw(self):
        """
        Redraws canvas, zooms.

        :return: None
        """
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

    def update_annotation_box(self):
        # Outer scope checks for csv path but not if image num in list of numbers
        curr_im = self.images[self.imageIndex]
        self.annotation_box.SetValue(
            'Ground Truth \n' + (self.annotated_map[curr_im] if curr_im in self.annotated_map.keys() else 'None'))

    def show_hide_annotations(self, event):
        if self.all_shown:
            annotation_dlg = wx.SingleChoiceDialog(self, message='Choose', caption='Choose',
                                                   choices=self.scorer.emotion_list())
            if annotation_dlg.ShowModal() == wx.ID_OK:
                self.show_annotations(annotation_type=annotation_dlg.GetStringSelection())
                self.all_shown = not self.all_shown
        else:
            self.AU_choices = self.make_full_au_choices()
            self.reorder_pics('Index')
            self.show_annotations_button.SetLabel('Show Only Annotated Frames')
            self.all_shown = not self.all_shown

    def show_annotations(self, annotation_type=None):
        if not annotation_type:
            self.AU_choices = list(self.annotated_map.keys())
        else:
            self.AU_choices = [i for i, val in self.annotated_map.items() if val == annotation_type]
        self.reorder_pics('Index')
        self.show_annotations_button.SetLabel('Show All Frames')

    def update_all(self):
        self.show_im()
        self.update_au_list()
        self.rewrite_text()
        if self.path_to_csv:
            self.update_annotation_box()

    def update_au_list(self):
        emotionDict = self.scorer.get_emotions(self.imageIndex)
        if emotionDict:
            emotionList = sorted(emotionDict.items(), key=lambda item: item[1],
                                 reverse=True)

            self.AU_List.Set(['Scores'] + ([emotion + ' = ' + str(value) for emotion, value in
                                            emotionList]) + ['Prevalence Score'] + [
                                 str(self.prevalence_score(emotionDict))])
        else:
            self.AU_List.Set(['None'])

    def make_full_au_choices(self):
        return [self.images[i] for i in self.scorer.emotions.keys()]

    @staticmethod
    def prevalence_score(emotionDict):
        reverse_emotions = {value: [x for x in emotionDict.keys() if emotionDict[x] == value] for value in
                            emotionDict.values()}
        max_value = max(reverse_emotions.keys())
        if len(reverse_emotions[max_value]) > 1:
            prevalence_score = 0
        else:
            prevalence_score = (
                (max_value ** 2) / np.sum([x * len(reverse_emotions[x]) for x in reverse_emotions.keys()]))
        return prevalence_score


def make_images():
    """
    Finds all png images in given directory.

    .. note:: Current working directory must be set before calling this function.

    :return: Sorted list of png images, 0
    """
    images = sorted(glob.glob('*.png'))
    imageIndex = 0
    return images, imageIndex


# csv_reader, from XMLTransformer
def csv_emotion_reader(csv_path):
    with open(csv_path, 'rt') as csv_file:
        image_map = {index - 1: row[1] for index, row in enumerate(csv.reader(csv_file)) if
                     index != 0}  # index -1 to compensate for first row offset
    return image_map


def au_name_dict():
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


if __name__ == '__main__':
    include_eyebrows = str(sys.argv[sys.argv.index('-eb') + 1]) == '1' if '-eb' in sys.argv else False
    csv_path = str(sys.argv[sys.argv.index('-csv') + 1]) if '-csv' in sys.argv else None
    directory = sys.argv[sys.argv.index('-d') + 1]
    app = wx.App(False)
    score = AUGui(None, wx.ID_ANY, "AuGUI", directory, include_eyebrows, csv_path)
    score.Show(True)
    app.MainLoop()
