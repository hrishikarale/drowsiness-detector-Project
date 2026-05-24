import numpy as np
import pygame

pygame.mixer.init(frequency=44100, size=-16, channels=2)

def generate_beep(frequency=880, duration=0.5, sample_rate=44100):
    # Create time values from 0 → duration
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Generate sine wave
    wave = np.sin(2 * np.pi * frequency * t)

    # Convert float wave to 16-bit audio
    wave = (wave * 32767).astype(np.int16)

    # Convert mono → stereo
    wave = np.column_stack((wave, wave))

    # Convert NumPy array to pygame Sound object
    return pygame.sndarray.make_sound(wave)


# Create beep sound
alarm_sound = generate_beep()

# Play once
alarm_sound.play(loops=-1)

input("Press Enter to stop...")