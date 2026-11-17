"""Select a sensor-map frontier without stopping MuJoCo's physics loop."""
import multiprocessing as mp
import time

import numpy as np


def _select(map_path, state, goal, expected_hash, connection):
    started = time.monotonic()
    try:
        from mission_navigation import choose_online_frontier
        from uav_continuum_whole_body_v2.mapping import MapSnapshot

        snapshot = MapSnapshot.load(map_path)
        if snapshot.map_hash != expected_hash:
            result = dict(status='map_changed_reject', kind='hold', waypoint=None,
                          information={'reason': 'map_changed_reject'})
        else:
            waypoint, kind, information = choose_online_frontier(snapshot, state, goal)
            result = dict(status='ready', kind=kind,
                          waypoint=None if waypoint is None else np.asarray(waypoint).tolist(),
                          information=information, map_hash=snapshot.map_hash)
        result['worker_wall_s'] = time.monotonic() - started
        connection.send(result)
    except Exception as error:
        connection.send(dict(status='worker_error_hold', kind='hold', waypoint=None,
                             information={'reason': type(error).__name__ + ': ' + str(error)},
                             worker_wall_s=time.monotonic() - started))
    finally:
        connection.close()


class AsyncFrontierSelector:
    def __init__(self):
        self.process = None
        self.connection = None
        self.request_state = None
        self.map_path = None
        self.map_hash = None
        self.started = None
        self.budget = None

    def request(self, map_path, state, goal, map_hash, budget=25.):
        if self.process is not None:
            raise RuntimeError('Frontier selection is already running')
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe(duplex=False)
        self.request_state = np.asarray(state, dtype=float).copy()
        self.map_path = map_path
        self.map_hash = map_hash
        self.started = time.monotonic()
        self.budget = float(budget)
        self.process = context.Process(target=_select,
            args=(str(map_path), self.request_state, np.asarray(goal, dtype=float), map_hash, child))
        self.process.start()
        child.close()

    def poll(self):
        if self.process is None:
            return None
        if self.connection.poll():
            result = self.connection.recv()
            result['selection_wall_s'] = time.monotonic() - self.started
            self.close()
            if result['selection_wall_s'] > self.budget:
                result.update(status='deadline_reject', kind='hold', waypoint=None)
                result['information'] = {'reason': 'frontier_selection_deadline'}
            return result
        if time.monotonic() - self.started > self.budget:
            elapsed = time.monotonic() - self.started
            self.close()
            return dict(status='deadline_hold', kind='hold', waypoint=None,
                        information={'reason': 'frontier_selection_deadline'},
                        selection_wall_s=elapsed)
        if not self.process.is_alive():
            self.close()
            return dict(status='worker_exited_hold', kind='hold', waypoint=None,
                        information={'reason': 'frontier_worker_exited_without_result'})
        return None

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=.05)
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
