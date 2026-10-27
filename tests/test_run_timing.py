"""Deterministic checks for displayed time estimates; no model fitting needed."""
import unittest
from unittest.mock import patch

from pyml_workbench.run_timing import RunTiming, SearchTimingEstimator


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.clock = patch('pyml_workbench.run_timing.time.monotonic', side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def test_actions_have_initial_estimates_and_overrun_keeps_estimate(self):
        for action in ('train', 'wizard_batch', 'freeze', 'test', 'export', 'inspect',
                       'restore', 'inference', 'batch-finalize', 'batch-freeze', 'unknown'):
            timing = RunTiming(action)
            timing.start()
            self.assertGreater(timing.remaining_seconds, 0)
            self.assertIn('初始', timing.estimate_basis)
            self.assertTrue(timing.stage_label)
            self.now += 1000
            self.assertTrue(timing.reference_exceeded)
            self.assertGreater(timing.remaining_seconds, 0)
            self.assertIsNotNone(timing.expected_finish_local)

    def test_stage_resets_only_stage_time_and_progress(self):
        timing = RunTiming('train')
        timing.start()
        self.now += 10
        timing.update_progress(1, 3)
        timing.set_stage('正在计算验证指标', 20)
        self.assertEqual(timing.elapsed_seconds, 10)
        self.assertEqual(timing.stage_elapsed_seconds, 0)
        self.assertEqual(timing.remaining_seconds, 20)
        self.assertEqual(timing.completed_work, 0)
        self.assertIsNone(timing.total_work)
        self.now += 5
        timing.set_stage('正在计算验证指标')
        self.assertEqual(timing.stage_elapsed_seconds, 5)

    def test_progress_samples_are_anchored_and_not_remeasured_every_tick(self):
        timing = RunTiming('train')
        timing.start()
        self.now += 10
        timing.update_progress(1, 3)
        self.assertEqual(timing.remaining_seconds, 20)
        self.now += 5
        timing.update_progress(1, 3)
        self.assertEqual(timing.remaining_seconds, 15)
        self.now += 15
        timing.update_progress(2, 3)
        self.assertEqual(timing.remaining_seconds, 15)

    def test_pause_freezes_active_time_and_finish_is_stable(self):
        timing = RunTiming('test')
        timing.start()
        self.now += 10
        timing.pause()
        remaining = timing.remaining_seconds
        self.now += 100
        self.assertEqual(timing.elapsed_seconds, 10)
        self.assertEqual(timing.remaining_seconds, remaining)
        self.assertIsNone(timing.expected_finish_local)
        timing.resume()
        self.now += 3
        ended, elapsed = timing.finish()
        self.assertEqual(elapsed, 13)
        self.now += 100
        self.assertEqual(timing.finish(), (ended, 13))
        self.assertEqual(timing.expected_finish_local, ended)
        self.assertEqual(timing.remaining_seconds, 0)

    def test_external_search_estimate_does_not_reset_elapsed(self):
        timing = RunTiming('wizard_batch')
        timing.start()
        self.now += 10
        timing.set_remaining_estimate(40, '实测1次')
        self.now += 5
        self.assertEqual(timing.elapsed_seconds, 15)
        self.assertEqual(timing.remaining_seconds, 35)
        self.assertEqual(timing.estimate_basis, '实测1次')


class SearchEstimateTests(unittest.TestCase):
    setUp = TimingTests.setUp
    @staticmethod
    def spec(fits=5, **kwargs):
        return dict(method='random', max_fits=fits, max_proposals=100,
                    timeout_seconds=3600, **kwargs)

    def event(self, estimator, name, count, *, job='a', trial=None, duration=None, status='complete'):
        data = dict(actual_fit_count=count, proposal_count=count, elapsed_seconds=self.now - 100)
        if duration is not None or trial is not None:
            data['record'] = dict(trial_id=trial, status=status, duration_seconds=duration)
        estimator.update_event(dict(job_id=job, name=name, trial_id=trial, data=data))

    def test_zero_one_two_samples_and_no_count_on_fit_start(self):
        estimator = SearchTimingEstimator()
        estimator.configure_job('a', self.spec())
        self.assertEqual(estimator.remaining_seconds(), 300)
        self.event(estimator, 'fit_started', 1, trial='t1')
        self.assertEqual(estimator.jobs['a'].completed_fit_count, 0)
        self.now += 10
        self.event(estimator, 'fit_completed', 1, trial='t1', duration=10)
        self.assertEqual(estimator.remaining_seconds(), 40)
        self.event(estimator, 'fit_started', 2, trial='t2')
        self.now += 30
        self.event(estimator, 'fit_completed', 2, trial='t2', duration=30)
        self.assertEqual(estimator.remaining_seconds(), 60)
        self.assertIn('2 次', estimator.estimate_basis)
        self.event(estimator, 'fit_completed', 2, trial='t2', duration=30)
        self.assertEqual(estimator.jobs['a'].fit_duration_count, 2)

    def test_failed_attempt_consumes_budget_but_does_not_calibrate(self):
        estimator = SearchTimingEstimator()
        estimator.configure_job('a', self.spec())
        self.event(estimator, 'fit_started', 1, trial='failure')
        self.now += 2
        self.event(estimator, 'fit_completed', 1, trial='failure', duration=2, status='failed')
        self.assertEqual(estimator.jobs['a'].actual_fit_count, 1)
        self.assertEqual(estimator.jobs['a'].fit_duration_count, 0)
        self.assertEqual(estimator.remaining_seconds(), 240)

    def test_delayed_duplicate_does_not_clear_next_active_fit(self):
        estimator = SearchTimingEstimator()
        estimator.configure_job('a', self.spec())
        self.event(estimator, 'fit_started', 1, trial='t1')
        self.now += 10
        self.event(estimator, 'fit_completed', 1, trial='t1', duration=10)
        self.event(estimator, 'fit_started', 2, trial='t2')
        self.now += 5
        self.event(estimator, 'fit_completed', 1, trial='t1', duration=10)
        self.assertEqual(estimator.remaining_seconds(), 35)
        self.assertEqual(estimator.jobs['a'].fit_duration_count, 1)

    def test_resume_uses_history_not_short_new_process_elapsed(self):
        estimator = SearchTimingEstimator()
        estimator.update_event(dict(job_id='a', name='search_started', data=dict(
            spec=self.spec(10), elapsed_seconds=80, actual_fit_count=8, proposal_count=8,
            resume_timing=dict(completed_fit_count=8, fit_duration_count=8, fit_duration_total=80))))
        self.assertEqual(estimator.remaining_seconds(), 20)

    def test_fast_samples_are_not_rounded_to_one_second_per_future_fit(self):
        estimator = SearchTimingEstimator()
        estimator.configure_job('a', self.spec(100), actual_fit_count=1, proposal_count=1,
                                completed_fit_count=1, fit_duration_count=1, fit_duration_total=.2)
        self.assertAlmostEqual(estimator.remaining_seconds(), 19.8)

    def test_parallel_keeps_slow_tail_and_queues_next_job(self):
        estimator = SearchTimingEstimator(parallel_workers=2)
        estimator.configure_job('a', self.spec(3), status='running', fit_duration_count=1, fit_duration_total=20)
        estimator.configure_job('b', self.spec(1), status='running', fit_duration_count=1, fit_duration_total=100)
        estimator.configure_job('c', self.spec(1), fit_duration_count=1, fit_duration_total=20)
        self.assertEqual(estimator.remaining_seconds(), 100)

    def test_grid_candidate_count_and_fixed_configuration(self):
        estimator = SearchTimingEstimator()
        estimator.configure_job('a', dict(method='grid', max_fits=20, max_proposals=100,
                                         space=dict(fields=dict(alpha=dict(values=[1, 2, 3])))))
        self.assertEqual(estimator.remaining_seconds(), 180)
        estimator.configure_job('a', dict(method='grid', max_fits=20, max_proposals=100, space=dict(fields={})))
        self.assertEqual(estimator.remaining_seconds(), 60)

    def test_current_fit_can_exceed_soft_time_budget(self):
        estimator = SearchTimingEstimator()
        spec = self.spec()
        spec['timeout_seconds'] = 60
        estimator.configure_job('a', spec)
        self.event(estimator, 'fit_started', 1, trial='t1')
        self.now += 70
        self.assertGreater(estimator.remaining_seconds(), 0)
        self.assertTrue(estimator.reference_exceeded)

    def test_pause_resume_and_finished_job_are_not_running(self):
        estimator = SearchTimingEstimator()
        estimator.configure_job('a', self.spec())
        self.event(estimator, 'fit_started', 1, trial='t1')
        self.now += 10
        self.event(estimator, 'search_paused', 1)
        remaining = estimator.remaining_seconds()
        self.now += 100
        self.assertTrue(estimator.is_paused)
        self.assertEqual(estimator.remaining_seconds(), remaining)
        self.event(estimator, 'search_resumed', 1)
        self.assertFalse(estimator.is_paused)
        self.assertEqual(estimator.remaining_seconds(), remaining)
        estimator.update_event(dict(job_id='a', name='search_finished', data=dict(result=dict(status='cancelled'))))
        self.assertFalse(estimator.is_paused)
        self.assertEqual(estimator.remaining_seconds(), 1)


if __name__ == '__main__':
    unittest.main()
