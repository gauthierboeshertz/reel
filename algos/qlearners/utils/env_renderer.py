# taken from https://github.com/jianxu305/openai-gym-docker/blob/main/example/Solving_CartPole_in_5_Lines.ipynb
# To get smooth animations
import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

def render_frames(frames,repeat=False, interval=40):
    def update_scene(num, frames, patch, time_text):
        patch.set_data(frames[num])
        text = f"frame: {num}"
        time_text.set_text(text)
        return patch, time_text

    fig = plt.figure()
    patch = plt.imshow(frames[0])
    ax = plt.gca()
    time_text = ax.text(0., 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch, time_text),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim
