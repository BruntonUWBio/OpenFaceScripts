"""
.. module:: au_gui
    :synopsis: A GUI for viewing a picture and the associated action units and emotion prediction.
"""

import csv
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import wx
from wx import media
from wx.lib.floatcanvas import NavCanvas

sys.path.append(os.path.abspath(__file__))
from scoring import OpenFaceScorer, AUScorer


class AUGui(wx.Frame):
    """
    Main GUI
    """

    def __init__(self,
                 parent,
                 frame_id,
                 name,
                 curr_directory,
                 path_to_csv=None):
        """
        Default constructor, creates a new AUGui

        :param parent: Inherited from wx.Frame
        :param frame_id: Unused, inherited from wx.Frame
        :param name: Unused, inherited from wx.Frame
        :param curr_directory: Directory with image files, AU files, etc.
        :param path_to_csv: Path to a CSV file with landmarks,
            in the form generated by FaceMapper
        """
        self.tmp_dir = 'tmp_video'
        self.path_to_csv = path_to_csv
        self.prominent_images = None
        self.skipping_index = 1
        self.fps_frac = 30  # Frames per second of video
        wx.Frame.__init__(self, parent, frame_id, name)
        os.chdir(curr_directory)
        self.images, self.imageIndex = make_images()

        self.image_map = None
        self.annotated_map = None
        self.all_shown = True

        if self.path_to_csv:
            self.image_map = csv_emotion_reader(path_to_csv)
            self.annotated_map = {
                self.images[index * self.fps_frac]: emotion
                for index, emotion in enumerate(
                    sorted(self.image_map.values()))
                if (index * self.fps_frac) < len(self.images)
            }  # Relies on image map only having one item per image

        self.AU_threshold = 0
        self.scorer = AUScorer.AUScorer(curr_directory)

        n_c = NavCanvas.NavCanvas(self, Debug=0, BackgroundColor="BLACK")
        self.Canvas = n_c.Canvas

        self.curr_emotions = []
        self.AU_choices = self.make_full_au_choices()
        self.AU_box = wx.BoxSizer(wx.VERTICAL)
        self.AU_List = wx.ListBox(
            self,
            wx.NewId(),
            style=wx.LC_REPORT | wx.SUNKEN_BORDER,
            name='List of Emotions',
            choices=self.curr_emotions)
        self.AU_box.Add(self.AU_List, 3, wx.EXPAND)

        if self.path_to_csv:
            self.annotation_box = wx.TextCtrl(
                self,
                wx.NewId(),
                value='N/A',
                style=wx.TE_READONLY | wx.TE_MULTILINE)
            self.AU_box.Add(self.annotation_box, 1, wx.EXPAND)
        self.pic_box = wx.ListBox(
            self,
            wx.NewId(),
            style=wx.LC_REPORT | wx.SUNKEN_BORDER,
            name='Pictures',
            choices=self.AU_choices)

        box = wx.BoxSizer(wx.HORIZONTAL)
        box.Add(self.pic_box, 1, wx.EXPAND)
        box.Add(n_c, 3, wx.EXPAND)
        box.Add(self.AU_box, 1, wx.EXPAND)

        botBox = wx.BoxSizer(wx.HORIZONTAL)
        self.order = 'Index'
        self.order_button = wx.Button(
            self, wx.NewId(), label='Order By Prominence')
        show_landmarksButton = wx.Button(
            self, wx.NewId(), label='Show/Hide Landmarks')
        self.au_text = wx.TextCtrl(
            self,
            wx.NewId(),
            value='N/A',
            style=wx.VSCROLL | wx.TE_READONLY | wx.TE_MULTILINE)

        show_vidButton = wx.Button(self, wx.NewId(), label='Show Video')

        botBox.Add(self.order_button, 1, wx.EXPAND)
        botBox.Add(show_landmarksButton, 1, wx.EXPAND)

        if self.path_to_csv:
            self.show_annotations_button = wx.Button(
                self, wx.NewId(), label='Show Annotated Frames')
            botBox.Add(self.show_annotations_button, 1, wx.EXPAND)
            self.Bind(
                wx.EVT_BUTTON,
                self.show_hide_annotations,
                id=self.show_annotations_button.GetId())
        botBox.Add(show_vidButton, 1, wx.EXPAND)
        botBox.Add(self.au_text, 4, wx.EXPAND)

        self.allBox = wx.BoxSizer(wx.VERTICAL)
        self.allBox.Add(box, 4, wx.EXPAND)
        self.allBox.Add(botBox, 1, wx.EXPAND)

        # -- Make Bindings --
        self.Bind(wx.EVT_LISTBOX, self.click_on_pic, id=self.pic_box.GetId())
        self.Bind(
            wx.EVT_LISTBOX, self.click_on_emotion, id=self.AU_List.GetId())
        self.Bind(
            wx.EVT_BUTTON, self.evt_reorder_pics, id=self.order_button.GetId())
        self.Bind(
            wx.EVT_BUTTON,
            self.show_landmarks,
            id=show_landmarksButton.GetId())
        self.Bind(wx.EVT_BUTTON, self.show_video, id=show_vidButton.GetId())

        self.SetSizer(self.allBox)
        self.Layout()
        self.bind_to_canvas()

        # Landmark stuff
        self.landmarks_exist = False
        self.landmarks_shown = False
        marked_pics_dir = os.path.join(curr_directory, 'labeled_frames/')

        if os.path.exists(marked_pics_dir):
            self.landmark_images = OpenFaceScorer.OpenFaceScorer.find_im_files(
                marked_pics_dir)

            if self.landmark_images:
                self.landmarks_exist = True

        self.update_all()

    def bind_to_canvas(self):
        """
        Binds necessary events to canvas.

        :return: None.
        """
        self.Canvas.Bind(wx.EVT_KEY_DOWN, self.on_key_press)

    def click_on_emotion(self, event):
        """
        Event handler for when an emotion is chosen from the list of emotions. Shows emotion templates.

        :param event: Unused
        :effects: Launches a :mod:`wx.Dialog` with the emotion templates.
        :return: None.
        """
        emote_template = self.scorer.emotion_templates()
        label = ''

        for emote in sorted(emote_template.keys()):
            label += '{0} : {1} \n'.format(emote, str(emote_template[emote]))
        self.pop_dialog('Emotion Templates', label)
        self.AU_List.Deselect(self.AU_List.GetSelection())

    def click_on_pic(self, event):
        """
        Event handler for when an image is chosen from the list of images. Updates so clicked pic is shown.

        :modifies: self.imageIndex to the index of the current image.
        :param event: Provides the name of the picture chosen.
        :return: None.
        """
        self.imageIndex = self.images.index(self.AU_choices[event.GetInt()])
        self.update_all()

    def show_landmarks(self, event):
        """
        If landmarks are available, either shows or hides face landmarks based on whether or not they are currently
        present: if shown, hides them and vice versa. If landmarks are unavailable, shows a dialog with that
        information.

        :param event: Unused.
        :return: None
        """

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
        """Changes currently displayed image to the image at self.imageIndex.

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
        frame_choice_str = 'Show Video around Frames'
        all_without_landmarks_choice_str = 'Show Entire Video (without landmarks)'
        all_choice_str = 'Show Entire Video (with landmarks)'
        choiceDlg = wx.SingleChoiceDialog(
            None,
            message='Please choose',
            caption='choose',
            choices=[
                frame_choice_str, all_without_landmarks_choice_str,
                all_choice_str
            ])

        if choiceDlg.ShowModal() == wx.ID_OK:
            selection = choiceDlg.GetStringSelection()

            if selection == frame_choice_str:
                self.show_video_around_frames()
            elif selection == all_without_landmarks_choice_str:
                self.show_entire_video(False)
            else:
                self.show_entire_video(True)

    def show_video_around_frames(self):
        """
        Displays media window with video with second before and after frame.

        :return: None

        .. note::Requires current working directory to be set to location of images
        """
        prev_next_frames = [
            self.images[i]
            for i in range(self.imageIndex - int(.5 * self.fps_frac),
                           self.imageIndex + int(.5 * self.fps_frac))
            if i in range(len(self.images))
        ]

        if os.path.exists(self.tmp_dir):
            self.rm_dir()
        os.mkdir(self.tmp_dir)

        for frame in prev_next_frames:
            shutil.copy(frame, self.tmp_dir)
        out_movie = os.path.join(self.tmp_dir, 'tmp_out.mp4')
        subprocess.Popen(
            "ffmpeg -r {2} -f image2 -s 1920x1080 -pattern_type glob -i '{0}' -b:v 2000k {1}"
            .format(
                os.path.join(self.tmp_dir, '*.png'), out_movie,
                str(self.fps_frac)),
            shell=True).wait()
        self.vid_panel(out_movie, tmp_dir=True)

    def show_entire_video(self, landmarks):
        """
        Shows entire video, with or without landmarks based on params.

        :requires: Cropped video saved as 'inter_out.mp4' in cwd and cropped video with landmarks saved as 'out.mp4' in cwd (both from CropAndOpenFace).
        :param landmarks: Whether landmarks should be displayed.
        :type landmarks: bool.
        :return: None
        """

        if not landmarks:
            self.vid_panel(out_movie='inter_out.mp4')
        else:
            self.vid_panel(out_movie='out.mp4')

    def vid_panel(self, out_movie, tmp_dir=False):
        frame = wx.Frame(None, wx.NewId(), "play audio and video files")
        panel1 = VidPanel(frame, wx.NewId(), self.scorer)
        frame.Show(True)
        panel1.do_load_file(out_movie)
        frame.Bind(wx.EVT_WINDOW_DESTROY, self.rm_dir)

    def rm_dir(self, event=None):
        """
        Removes the temporary directory created by this, if it exists.
        :param event: Unused.
        :return: None
        """

        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def evt_reorder_pics(self, event=None):
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
                prevalence_score(self.scorer.get_emotions(self.images.index(pic))) if self.images.index(
                    pic) in self.scorer.emotions.keys() else 0),
                reverse=True)
            self.order_button.SetLabel('Order by Index')
        else:
            raise ValueError('Unknown order type')
        self.order = order_type
        self.pic_box.Set(self.AU_choices)

    def rewrite_text(self):
        label = 'Current Image = ' + \
            self.images[self.imageIndex] + '\n\n' + 'AUs' + '\n'
        name_dict = au_name_dict()

        if self.scorer.presence_dict[self.imageIndex]:
            au_dict = self.scorer.presence_dict[self.imageIndex]
            au_dict_keys = sorted(au_dict.keys())

            for au in au_dict_keys:
                if 'c' in au:
                    r_label = au.replace('c', 'r')
                    au_int = return_num(au)
                    label += '{0} ({1}) = {2} \n'.format(
                        str(au_int), name_dict[au_int],
                        str(au_dict[r_label])
                        if r_label in au_dict_keys else 'Present')
        self.au_text.SetValue(label)

    def redraw(self):
        """
        Redraws canvas, zooms.

        :return: None
        """
        self.Canvas.Draw()
        self.Canvas.ZoomToBB()

    def on_key_press(self, event):
        """
        Event handler for key press. If the right arrow key is pressed, displays the next image in the directory (in
        lexicographically sorted order). If the left arrow key is pressed, displays the previous image in the
        directory.

        :param event: Keypress.
        :type event: wx.Event
        """
        keyCode = event.GetKeyCode()

        if keyCode == wx.WXK_RIGHT:
            while wx.GetKeyState(wx.WXK_RIGHT) and (
                    self.imageIndex + self.skipping_index) in range(
                        len(self.images)):
                self.imageIndex += self.skipping_index
                self.update_all()
        elif keyCode == wx.WXK_LEFT:
            while wx.GetKeyState(wx.WXK_LEFT) and (
                    self.imageIndex - self.skipping_index) in range(
                        len(self.images)):
                self.imageIndex -= self.skipping_index
                self.update_all()

    def update_annotation_box(self):
        # Outer scope checks for csv path but not if image num in list of numbers
        curr_im = self.images[self.imageIndex]
        self.annotation_box.SetValue('Ground Truth \n' +
                                     (self.annotated_map[curr_im] if curr_im in
                                      self.annotated_map.keys() else 'None'))

    def show_hide_annotations(self, event):
        if self.all_shown:
            annotation_dlg = wx.SingleChoiceDialog(
                self,
                message='Choose',
                caption='Choose',
                choices=self.scorer.emotion_list() + (['All', 'Neutral']))

            if annotation_dlg.ShowModal() == wx.ID_OK:
                selection = annotation_dlg.GetStringSelection()
                type = None if selection == 'All' else selection
                self.show_annotations(annotation_type=type)
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
            self.AU_choices = [
                i for i, val in self.annotated_map.items()
                if val == annotation_type
            ]
        self.reorder_pics('Index')
        self.show_annotations_button.SetLabel('Show All Frames')

    def update_all(self):
        self.show_im()
        update_emotion_list(self.scorer, self.AU_List, self.imageIndex)
        self.rewrite_text()

        if self.path_to_csv:
            self.update_annotation_box()

    def make_full_au_choices(self):
        return [self.images[i] for i in self.scorer.emotions.keys()]


class VidPanel(wx.Panel):
    """
    Panel for showing video. Code originally from
    https://www.daniweb.com/programming/software-development/code/216704/wxpython-plays-audio-and-video-files.
    """

    def __init__(self, parent, id, scorer):
        """
        Default constructor.

        :param parent: wx.Frame object.
        :param id: ID
        """
        # self.log = log
        wx.Panel.__init__(
            self, parent, id, style=wx.TAB_TRAVERSAL | wx.CLIP_CHILDREN)
        # Create some controls
        try:
            self.mc = wx.media.MediaCtrl(self, style=wx.SIMPLE_BORDER)
        except NotImplementedError as e:
            self.Destroy()
            raise e
        self.parent = parent
        loadButton = wx.Button(self, -1, "Load File")
        self.Bind(wx.EVT_BUTTON, self.onLoadFile, loadButton)

        playButton = wx.Button(self, -1, "Play")
        self.Bind(wx.EVT_BUTTON, self.onPlay, playButton)

        pauseButton = wx.Button(self, -1, "Pause")
        self.Bind(wx.EVT_BUTTON, self.onPause, pauseButton)

        stopButton = wx.Button(self, -1, "Stop")
        self.Bind(wx.EVT_BUTTON, self.onStop, stopButton)

        slider = wx.Slider(self, -1, 0, 0, 0, size=wx.Size(300, -1))
        self.slider = slider
        self.Bind(wx.EVT_SLIDER, self.onSeek, slider)

        self.scorer = scorer

        self.st_file = wx.StaticText(
            self, -1, ".mid .mp3 .wav .au .avi .mpg", size=(200, -1))
        self.st_size = wx.StaticText(self, -1, size=(100, -1))
        self.st_len = wx.StaticText(self, -1, size=(100, -1))
        self.st_pos = wx.StaticText(self, -1, size=(100, -1))

        self.emotion_list = AUScorer.emotion_list()
        full_au_list = self.emotion_list + \
            (['Best Score', '', 'Prominence', ''])
        self.emotion_texts = [
            wx.StaticText(
                self, wx.NewId(), "{0}".format(emotion), size=(100, -1))
            for emotion in full_au_list
        ]
        self.blank_gauges = [
            wx.Gauge(self, wx.NewId(), size=(100, -1))
            for emotion in full_au_list
        ]
        self.blank_gauges[len(self.emotion_list)] = wx.StaticText(
            self, wx.NewId(), size=(100, -1))
        self.blank_gauges[len(self.emotion_list) + 1] = wx.StaticText(
            self, wx.NewId(), size=(100, -1))

        # setup the button/label layout using a sizer
        sizer = wx.GridBagSizer(6, 6)
        sizer.Add(loadButton, (1, 1))
        sizer.Add(playButton, (2, 1))
        sizer.Add(pauseButton, (3, 1))
        sizer.Add(stopButton, (4, 1))
        sizer.Add(self.st_file, (1, 2))
        sizer.Add(self.st_size, (2, 2))
        sizer.Add(self.st_len, (3, 2))
        sizer.Add(self.st_pos, (4, 2))

        for index, emotionText in enumerate(self.emotion_texts):
            sizer.Add(emotionText, (index + 1, 5))

        for index, blank_gauge in enumerate(self.blank_gauges):
            sizer.Add(blank_gauge, (index + 1, 6))

        sizer.Add(self.mc, (6, 1), span=(4, 4))  # for .avi .mpg video files
        self.SetSizer(sizer)
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.onTimer)
        self.timer.Start(100)

    def onLoadFile(self, evt):
        """
        Event handler for loading a file. Launches a dialog for picking a file to load.

        :param evt: Unused.
        """
        dlg = wx.FileDialog(
            self,
            message="Choose a media file",
            defaultDir=os.getcwd(),
            defaultFile="",
            style=wx.FD_OPEN | wx.FD_CHANGE_DIR)

        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.do_load_file(path)
        dlg.Destroy()

    def do_load_file(self, path):
        if not self.mc.Load(path):
            wx.MessageBox("Unable to load %s: Unsupported format?" % path,
                          "ERROR", wx.ICON_ERROR | wx.OK)
        else:
            folder, filename = os.path.split(path)
            self.st_file.SetLabel('%s' % filename)
            self.parent.SetSize(self.GetBestSize())
            self.GetSizer().Layout()
            self.slider.SetRange(0, self.mc.Length())
            self.mc.Play()

    def onPlay(self, evt):
        """
        Event handler for playing a video.

        :param evt: Unused.
        :return: None
        """
        self.mc.Play()
        self.setAllGauges()

    def onPause(self, evt):
        """
        Event handler for pausing a video.

        :param evt: Unused.
        :return: None
        """
        self.mc.Pause()
        self.setAllGauges()

    def onStop(self, evt):
        """
        Event handler for stopping a video.

        :param evt: Unused.
        :return: None
        """
        self.mc.Stop()
        self.setAllGauges()

    def onSeek(self, evt):
        """
        Event handler for seeking a video.

        :param evt: Unused.
        :return: None
        """
        offset = self.slider.GetValue()
        self.mc.Seek(offset)
        self.setAllGauges()

    def onTimer(self, evt):
        """
        Event handler for updating timers.

        :param evt: Unused.
        :return: None
        """
        offset = self.mc.Tell()
        self.slider.SetValue(offset)
        self.st_size.SetLabel('size: %s ms' % self.mc.Length())
        self.st_len.SetLabel('( %d seconds )' % (self.mc.Length() / 1000))
        self.st_pos.SetLabel('position: %d ms' % offset)

    def setAllGauges(self):
        offset = self.mc.Tell()
        frame = int(offset / (1000 / 30))
        emotionDict = self.scorer.get_emotions(frame)

        for emotion in self.emotion_list:
            currGauge = self.blank_gauges[self.emotion_list.index(emotion)]

            if emotion in emotionDict.keys():
                currGauge.SetValue((emotionDict[emotion] / 5) * 100)
            else:
                currGauge.SetValue(0)

        if emotionDict:
            self.blank_gauges[len(self.emotion_list)].SetLabel(
                sorted(
                    emotionDict.items(),
                    key=lambda item: item[1],
                    reverse=True)[0][0])
        else:
            self.blank_gauges[len(self.emotion_list)].SetLabel('N/A')
        self.blank_gauges[len(self.blank_gauges) - 1].SetValue(
            (prevalence_score(emotionDict) / 5) * 100)


def prevalence_score(emotionDict):
    """
    Calculate a prevalence score for the max emotion in a emotion dictionary

    :param emotionDict: Dictionary mapping one of the basic emotions to its corresponding score (calculated by AUScorer)
    :return: Score calculated using both value of highest value emotion as well as how prevalent that emotion is
    """
    reverse_emotions = AUScorer.reverse_emotions(emotionDict)

    if reverse_emotions:
        max_value = max(reverse_emotions.keys())
        # if len(reverse_emotions[max_value]) > 1:
        #    score = 0
        # else:
        score = ((max_value**2) / np.sum(
            [x * len(reverse_emotions[x]) for x in reverse_emotions.keys()]))
    else:
        score = 0

    return score


def make_images():
    """
    Finds all png images in given directory.

    :return: Sorted list of png images, 0.

    .. note:: Current working directory must be set before calling this function.
    """
    images = sorted(glob.glob('*.png'))
    imageIndex = 0

    return images, imageIndex


# csv_reader, from XMLTransformer
def csv_emotion_reader(csv_path):
    """
    Reads emotions from csv file.

    :param csv_path: Path to csv file.
    :return: Dictionary mapping filenames (from csv) to emotions labelled in filename.
    """
    with open(csv_path, 'rt') as csv_file:
        image_map = {
            index - 1: row[1]
            for index, row in enumerate(csv.reader(csv_file)) if index != 0
        }  # index -1 to compensate for first row offset

    return image_map


def au_name_dict():
    """
    Creates a mapping between Action Unit numbers and the associated names

    :return: Dictionary with mapping
    """

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


def update_emotion_list(scorer, emotion_list, index):
    """
    Updates list of emotions
    :param scorer: AU scorer
    :type scorer: AUScorer.AUScorer
    :param emotion_list: List of emotions to update
    :type emotion_list: wx.ListBox
    :param index: image index
    :return: None
    """
    emotionDict = scorer.get_emotions(index)

    if emotionDict:
        emotionList = sorted(
            emotionDict.items(), key=lambda item: item[1], reverse=True)
        # emotionList = sorted(emotionDict.items()) #other option for sorting

        emotion_list.Set(['Scores'] + (
            [emotion + ' = ' + str(value)
             for emotion, value in emotionList]) + ['Prevalence Score'] +
                         [str(prevalence_score(emotionDict))])
    else:
        emotion_list.Set(['None'])


if __name__ == '__main__':
    include_eyebrows = str(sys.argv[sys.argv.index('-eb') +
                                    1]) == '1' if '-eb' in sys.argv else False
    # csv_path = str(sys.argv[sys.argv.index('-csv') + 1]) if '-csv' in sys.argv else None
    # directory = sys.argv[sys.argv.index('-d') + 1]
    app = wx.App(False)

    if '-d' in sys.argv:
        dir_dlg = wx.DirDialog(None, message='Please select a directory')
        directory = dir_dlg.GetPath() if dir_dlg.ShowModal(
        ) == wx.ID_OK else None

    if '-csv' in sys.argv:
        csv_dialog = wx.FileDialog(None, message="Please select a csv file.")
        csv_path = csv_dialog.GetPath() if csv_dialog.ShowModal(
        ) == wx.ID_OK else None
    else:
        csv_path = None
    score = AUGui(None, wx.ID_ANY, "AuGUI", directory, include_eyebrows,
                  csv_path)
    score.Show(True)
    app.MainLoop()
