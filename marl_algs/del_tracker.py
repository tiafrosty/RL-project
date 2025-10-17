
from collections import deque

##########33 my brand new delay tracker ################33
##################################################
class DelayTracker:
    """
    Tracks queueing delay measured in slots:
    delay = (service_slot - arrival_slot) for each served packet.
    """
    def __init__(self, n_agents: int):
        self.arrival_times = [deque() for _ in range(n_agents)]
        self.avg_delay_per_slot = []
        self._slot_delay_sum = 0
        self._number_of_signals_served = 0
        self.total_delay_sum = 0
        self.total_served = 0

    # slot cycle
    def start_slot(self):
        self._slot_delay_sum = 0
        self._number_of_signals_served = 0

    def end_slot(self):
        avg = (self._slot_delay_sum / self._number_of_signals_served) if self._number_of_signals_served > 0 else 0.0
        self.avg_delay_per_slot.append(avg)
        return avg

    def record_arrival(self, agent_idx: int, arrival_slot: int):
        """when a packet arrives and is enqueued"""
        self.arrival_times[agent_idx].append(arrival_slot)

    def record_service(self, agent_idx: int, service_slot: int):
        """
        Returns the delay for this served packet, or None if queue was empty (shouldnt happen i serve w with B>0).
        """
        if self.arrival_times[agent_idx]:
            t_arr = self.arrival_times[agent_idx].popleft()
            d = service_slot - t_arr   # 0 if arrived and served in same slot
            self._slot_delay_sum += d
            self.total_delay_sum += d
            self._number_of_signals_served += 1
            self.total_served += 1
            return d
        return None

    def overall_avg_delay(self):
        return (self.total_delay_sum / self.total_served) if self.total_served > 0 else float('nan')

#######################################################33
##########################333
