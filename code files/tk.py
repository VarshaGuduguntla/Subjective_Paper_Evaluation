from tkinter import *
import sys
import var
sys.path.append("eval.py")

class App(Frame):
    def run_script(self):
        sys.stdout = self
        try:
            del (sys.modules["Descriptive Answer Analysis"])
        except:
            pass
            import eval

    def build_widgets(self):
        self.text1 = Text(self)
        self.text1.pack(side=TOP)
        self.button = Button(self)
        self.button["text"] = "Run"
        self.button["command"] = self.run_script
        self.button.pack(side=TOP)

    def write(self, txt):
        self.text1.insert(INSERT, txt)

    def flush(self):
        pass

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.build_widgets()

    def show_entry_fields(self):
        print("Enter student filename: %s" % (var.ans.get()))
        var.ans = var.ans.get()

    def on_button(self):
        print(self.entry.get())

    def show_entry_fields(self):
        print("Enter Answerkey name: %s" % (var.modelans.get()))
        var.modelans = var.modelans.get()

root = Tk()
root.title("Descriptive paper evaluation")
app = App(master=root)
app.mainloop()

