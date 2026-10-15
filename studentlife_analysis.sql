-- Development participants only. All outputs stay in data/private_studentlife/sql/.
-- Query 1: JOIN feature records to per-channel observation counts.
SELECT p.student_id, c.channel, c.observed_days, c.window_days,
       c.window_days - c.observed_days AS days_without_recorded_observations,
       ROUND(100.0 * c.observed_days / c.window_days, 1) AS observed_day_percent
FROM participants AS p
JOIN coverage AS c ON c.student_id = p.student_id
ORDER BY p.student_id, c.channel;

-- Query 2: Missing prepared values are distinct from missing raw observations.
SELECT feature, COUNT(*) AS participants,
       COUNT(value) AS observed_values,
       SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) AS missing_values
FROM feature_values
GROUP BY feature
ORDER BY feature;

-- Query 3: Descriptive feature statistics; no target-based model selection.
SELECT feature, unit, COUNT(value) AS observed_values,
       ROUND(AVG(value), 3) AS mean,
       ROUND(MIN(value), 3) AS minimum,
       ROUND(MAX(value), 3) AS maximum
FROM feature_values
GROUP BY feature, unit
ORDER BY feature;

-- Query 4: Filter channels with at least one development participant
-- having observations on fewer than half the window days.
SELECT channel, COUNT(*) AS participants,
       ROUND(AVG(observed_days), 2) AS mean_observed_days,
       SUM(CASE WHEN observed_days * 2 < window_days THEN 1 ELSE 0 END)
           AS participants_below_half_window
FROM coverage
GROUP BY channel
HAVING SUM(CASE WHEN observed_days * 2 < window_days THEN 1 ELSE 0 END) > 0
ORDER BY channel;
