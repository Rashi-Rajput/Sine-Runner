import json
from stable_baselines3.common.callbacks import BaseCallback
from .evaluate_rl import evaluate, VALIDATION_SEEDS


class TrainingCallback(BaseCallback):
    def __init__(self, output, frequency, env_kwargs):
        super().__init__()
        self.output, self.frequency, self.env_kwargs = output, frequency, env_kwargs
        self.next_eval, self.best = frequency, (-1, float('-inf'))

    def _on_training_start(self):
        self.next_eval = self.num_timesteps + self.frequency
        # Establish the starting policy's score, including when resuming.
        self.assess()

    def _on_step(self):
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done:
                for key in ('success', 'distance', 'score', 'coins', 'crashes', 'hearts', 'episode_length', 'episode_return'):
                    self.logger.record_mean('episodes/' + key, float(info[key]))
        if self.num_timesteps >= self.next_eval:
            self.assess()
            self.next_eval = self.num_timesteps + self.frequency
        return True

    def assess(self):
        summary = evaluate(self.model, VALIDATION_SEEDS, **self.env_kwargs)['summary']
        rank = summary['completion_rate'], summary['episode_return']['mean']
        for key, value in summary.items():
            self.logger.record('eval/' + key, value['mean'] if isinstance(value, dict) else value)
        self.model.save(self.output / 'latest_model')
        if rank > self.best:
            self.best = rank
            self.model.save(self.output / 'best_model')
            (self.output / 'best_score.json').write_text(json.dumps(
                dict(step=self.num_timesteps, summary=summary), indent=2))
        with (self.output / 'evaluations.jsonl').open('a') as log:
            log.write(json.dumps(dict(step=self.num_timesteps, summary=summary)) + '\n')
        self.logger.dump(self.num_timesteps)

    def _on_training_end(self):
        self.assess()
