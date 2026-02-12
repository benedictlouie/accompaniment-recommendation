from abc import ABC, abstractmethod


class BaseChordEngine(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def process_beat(self, notes_played, beat_start_time, beat_index):
        """
        Called once per beat.

        Returns:
            chord_name (str or None),
            duration_seconds (float)
        """
        pass
