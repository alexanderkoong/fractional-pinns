#!/usr/bin/env python3
"""
Data loader for tweet engagement time series.
Converts CSV data into format expected by the PINN.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class LoadStats:
    """Statistics from robust data loading."""
    total_tweets: int
    kept_complete: int
    kept_interpolated: int
    excluded_no_start_hour: int
    excluded_large_gaps: int
    excluded_zero_engagement: int
    excluded_flat_engagement: int
    interpolation_details: dict  # {tweet_id: [interpolated_hours]}

    @property
    def total_kept(self) -> int:
        return self.kept_complete + self.kept_interpolated

    @property
    def total_excluded_engagement(self) -> int:
        return self.excluded_zero_engagement + self.excluded_flat_engagement

    @property
    def retention_rate(self) -> float:
        return self.total_kept / self.total_tweets if self.total_tweets > 0 else 0.0

    def __repr__(self):
        return (
            f"LoadStats(\n"
            f"  total_tweets={self.total_tweets},\n"
            f"  kept_complete={self.kept_complete} ({100*self.kept_complete/max(1,self.total_tweets):.1f}%),\n"
            f"  kept_interpolated={self.kept_interpolated} ({100*self.kept_interpolated/max(1,self.total_tweets):.1f}%),\n"
            f"  excluded_no_start_hour={self.excluded_no_start_hour} ({100*self.excluded_no_start_hour/max(1,self.total_tweets):.1f}%),\n"
            f"  excluded_large_gaps={self.excluded_large_gaps} ({100*self.excluded_large_gaps/max(1,self.total_tweets):.1f}%),\n"
            f"  excluded_zero_engagement={self.excluded_zero_engagement} ({100*self.excluded_zero_engagement/max(1,self.total_tweets):.1f}%),\n"
            f"  excluded_flat_engagement={self.excluded_flat_engagement} ({100*self.excluded_flat_engagement/max(1,self.total_tweets):.1f}%),\n"
            f"  retention_rate={100*self.retention_rate:.1f}%\n"
            f")"
        )


def _find_consecutive_gaps(missing_hours: List[int]) -> List[List[int]]:
    """
    Find groups of consecutive missing hours.

    Args:
        missing_hours: Sorted list of missing hour indices

    Returns:
        List of lists, each containing consecutive missing hours
    """
    if not missing_hours:
        return []

    gaps = []
    current_gap = [missing_hours[0]]

    for i in range(1, len(missing_hours)):
        if missing_hours[i] == missing_hours[i-1] + 1:
            current_gap.append(missing_hours[i])
        else:
            gaps.append(current_gap)
            current_gap = [missing_hours[i]]
    gaps.append(current_gap)

    return gaps


def _linear_interpolate(values: np.ndarray, missing_indices: List[int]) -> np.ndarray:
    """
    Perform linear interpolation for missing values.

    Args:
        values: Array with NaN at missing positions
        missing_indices: List of indices to interpolate

    Returns:
        Array with interpolated values
    """
    result = values.copy()

    for idx in missing_indices:
        # Find the nearest known values on both sides
        left_idx = idx - 1
        right_idx = idx + 1

        # Handle edge cases (shouldn't happen if we require hour 0)
        while left_idx >= 0 and np.isnan(result[left_idx]):
            left_idx -= 1
        while right_idx < len(result) and np.isnan(result[right_idx]):
            right_idx += 1

        if left_idx >= 0 and right_idx < len(result):
            # Linear interpolation
            left_val = result[left_idx]
            right_val = result[right_idx]
            span = right_idx - left_idx
            result[idx] = left_val + (right_val - left_val) * (idx - left_idx) / span
        elif left_idx >= 0:
            # Extrapolate from left (shouldn't happen with our design)
            result[idx] = result[left_idx]
        elif right_idx < len(result):
            # Extrapolate from right (shouldn't happen with our design)
            result[idx] = result[right_idx]

    return result


def load_tweet_data_robust(
    snapshots_path: str,
    metric: str = 'likes',
    start_hour: int = 0,
    end_hour: int = 47,
    max_gap_to_interpolate: int = 2,
    normalize: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str], LoadStats]:
    """
    Load tweet snapshots with robust handling of missing data.

    Strategy:
    1. Use hours [start_hour, end_hour] (default 0-47, giving 48 timesteps)
    2. Require start_hour data (exclude tweets missing it)
    3. For middle-hour gaps: interpolate if <= max_gap_to_interpolate, else exclude

    Note: Fractional hours_since_post values are bucketed to integers using floor.
    E.g., 0.017 -> hour 0, 1.334 -> hour 1, 47.94 -> hour 47

    Args:
        snapshots_path: Path to CSV file with tweet snapshots
        metric: Engagement metric ('likes', 'retweets', 'replies', 'quotes', 'total')
        start_hour: First hour to include (default 0)
        end_hour: Last hour to include (default 47)
        max_gap_to_interpolate: Maximum consecutive missing hours to interpolate (default 2)
        normalize: If True, normalize engagement to [0, 1] per tweet
        verbose: If True, print detailed statistics

    Returns:
        t: (num_tweets, num_timesteps, 1) - time points
        c: (num_tweets, num_timesteps, 1) - engagement values
        post_ids: List of post IDs for kept tweets
        stats: LoadStats object with detailed statistics
    """
    # Load raw data
    df = pd.read_csv(snapshots_path)

    # Bucket fractional hours to integers (floor)
    # E.g., 0.017 -> 0, 1.334 -> 1, 47.94 -> 47
    df['hour_bucket'] = df['hours_since_post'].apply(lambda x: int(x))

    # Expected hours
    expected_hours = list(range(start_hour, end_hour + 1))
    num_timesteps = len(expected_hours)

    # Get unique post IDs
    all_post_ids = df['post_id'].unique()
    total_tweets = len(all_post_ids)

    if verbose:
        print(f"Loading data with robust missing data handling...")
        print(f"  Time window: hours {start_hour}-{end_hour} ({num_timesteps} timesteps)")
        print(f"  Max interpolation gap: {max_gap_to_interpolate} hours")
        print(f"  Total tweets in dataset: {total_tweets}")
        print(f"  Note: Fractional hours bucketed to integers (e.g., 1.017 -> 1)")

    # Tracking statistics
    kept_complete = 0
    kept_interpolated = 0
    excluded_no_start_hour = 0
    excluded_large_gaps = 0
    interpolation_details = {}

    # Lists to build final arrays
    kept_post_ids = []
    kept_t = []
    kept_c = []

    for pid in all_post_ids:
        tweet_df = df[df['post_id'] == pid].copy()

        # Get available hour buckets for this tweet
        available_hours = set(tweet_df['hour_bucket'].values)

        # Check 1: Does it have the start hour?
        if start_hour not in available_hours:
            excluded_no_start_hour += 1
            continue

        # Check 2: What hours in [start_hour, end_hour] are missing?
        missing_hours = [h for h in expected_hours if h not in available_hours]

        if not missing_hours:
            # Complete data - no interpolation needed
            # Filter to expected hour range and keep one snapshot per hour bucket
            tweet_df = tweet_df[tweet_df['hour_bucket'].isin(expected_hours)]
            # If multiple snapshots in same hour bucket, keep the first one
            tweet_df = tweet_df.sort_values('hours_since_post').drop_duplicates(subset='hour_bucket', keep='first')
            tweet_df = tweet_df.sort_values('hour_bucket')

            if metric == 'total':
                engagement = (tweet_df['likes'] + tweet_df['retweets'] +
                             tweet_df['replies'] + tweet_df['quotes']).values
            else:
                engagement = tweet_df[metric].values

            kept_post_ids.append(pid)
            kept_t.append(np.array(expected_hours, dtype=float))
            kept_c.append(engagement.astype(float))
            kept_complete += 1
            continue

        # Check 3: Can we interpolate the gaps?
        gap_groups = _find_consecutive_gaps(sorted(missing_hours))
        max_gap_size = max(len(g) for g in gap_groups)

        if max_gap_size > max_gap_to_interpolate:
            # Gap too large - exclude this tweet
            excluded_large_gaps += 1
            continue

        # Interpolate the missing values
        # First, filter to expected hour range and deduplicate
        tweet_df = tweet_df[tweet_df['hour_bucket'].isin(expected_hours)]
        tweet_df = tweet_df.sort_values('hours_since_post').drop_duplicates(subset='hour_bucket', keep='first')
        tweet_df = tweet_df.sort_values('hour_bucket')

        # Create full arrays with NaN for missing
        full_engagement = np.full(num_timesteps, np.nan)

        for _, row in tweet_df.iterrows():
            hour_idx = int(row['hour_bucket']) - start_hour
            if metric == 'total':
                val = row['likes'] + row['retweets'] + row['replies'] + row['quotes']
            else:
                val = row[metric]
            full_engagement[hour_idx] = float(val)

        # Get indices of missing values (relative to expected_hours)
        missing_indices = [h - start_hour for h in missing_hours]

        # Interpolate
        interpolated_engagement = _linear_interpolate(full_engagement, missing_indices)

        # Verify no NaN remains
        if np.any(np.isnan(interpolated_engagement)):
            excluded_large_gaps += 1
            continue

        kept_post_ids.append(pid)
        kept_t.append(np.array(expected_hours, dtype=float))
        kept_c.append(interpolated_engagement)
        kept_interpolated += 1
        interpolation_details[pid] = missing_hours

    # === ENGAGEMENT FILTER: Remove zero and flat engagement tweets ===
    # This must happen before normalization to check actual values
    excluded_zero_engagement = 0
    excluded_flat_engagement = 0

    filtered_post_ids = []
    filtered_t = []
    filtered_c = []

    for i in range(len(kept_post_ids)):
        engagement = kept_c[i]
        max_val = engagement.max()
        min_val = engagement.min()

        if max_val == 0:
            # Zero engagement throughout - exclude
            excluded_zero_engagement += 1
            continue
        elif max_val == min_val:
            # Flat engagement (no change from start to end) - exclude
            excluded_flat_engagement += 1
            continue

        # Tweet has actual growth - keep it
        filtered_post_ids.append(kept_post_ids[i])
        filtered_t.append(kept_t[i])
        filtered_c.append(kept_c[i])

    # Update counts after engagement filtering
    # Adjust kept_complete and kept_interpolated proportionally
    original_kept = kept_complete + kept_interpolated
    filtered_kept = len(filtered_post_ids)
    if original_kept > 0:
        keep_ratio = filtered_kept / original_kept
        # We can't know exactly which were complete vs interpolated, so adjust proportionally
        kept_complete = int(kept_complete * keep_ratio)
        kept_interpolated = filtered_kept - kept_complete

    kept_post_ids = filtered_post_ids
    kept_t = filtered_t
    kept_c = filtered_c

    # Build statistics
    stats = LoadStats(
        total_tweets=total_tweets,
        kept_complete=kept_complete,
        kept_interpolated=kept_interpolated,
        excluded_no_start_hour=excluded_no_start_hour,
        excluded_large_gaps=excluded_large_gaps,
        excluded_zero_engagement=excluded_zero_engagement,
        excluded_flat_engagement=excluded_flat_engagement,
        interpolation_details=interpolation_details
    )

    if verbose:
        print(f"\n{stats}")

    if not kept_post_ids:
        raise ValueError("No tweets passed the filtering criteria!")

    # Convert to numpy arrays
    num_kept = len(kept_post_ids)
    t = np.zeros((num_kept, num_timesteps, 1))
    c = np.zeros((num_kept, num_timesteps, 1))

    for i in range(num_kept):
        # Normalize time to [0, 1]
        t[i, :, 0] = kept_t[i] / kept_t[i].max()
        c[i, :, 0] = kept_c[i]

    # Normalize engagement to [0, 1] per tweet
    if normalize:
        for i in range(num_kept):
            max_val = c[i, :, 0].max()
            if max_val > 0:
                c[i, :, 0] = c[i, :, 0] / max_val

    if verbose:
        print(f"\nFinal data shape - t: {t.shape}, c: {c.shape}")
        print(f"Time range: [{t.min():.3f}, {t.max():.3f}]")
        print(f"Engagement range: [{c.min():.3f}, {c.max():.3f}]")

    return t, c, kept_post_ids, stats


def load_tweet_data(snapshots_path, metric='likes', normalize=True):
    """
    Load tweet snapshots and reshape for PINN training.

    Args:
        snapshots_path: Path to sample_snapshots.csv
        metric: Which engagement metric to use ('likes', 'retweets', 'replies', 'quotes')
                Can also be 'total' to sum all metrics
        normalize: If True, normalize engagement to [0, 1] range per tweet

    Returns:
        t: (num_tweets, num_timesteps, 1) - time points
        c: (num_tweets, num_timesteps, 1) - engagement values
        post_ids: list of post IDs for reference
    """
    # Load data
    df = pd.read_csv(snapshots_path)

    # Get unique post IDs
    post_ids = df['post_id'].unique()
    num_tweets = len(post_ids)

    # Determine number of timesteps (assume all tweets have same count)
    num_timesteps = len(df[df['post_id'] == post_ids[0]])

    print(f"Loading {num_tweets} tweets with {num_timesteps} timesteps each")

    # Initialize arrays
    t = np.zeros((num_tweets, num_timesteps, 1))
    c = np.zeros((num_tweets, num_timesteps, 1))

    # Fill arrays
    for i, pid in enumerate(post_ids):
        tweet_df = df[df['post_id'] == pid].sort_values('hours_since_post')

        # Time (in hours, normalized to [0, 1] range for stability)
        hours = tweet_df['hours_since_post'].values
        t[i, :, 0] = hours / hours.max()  # Normalize to [0, 1]

        # Engagement metric
        if metric == 'total':
            engagement = (tweet_df['likes'] + tweet_df['retweets'] +
                         tweet_df['replies'] + tweet_df['quotes']).values
        else:
            engagement = tweet_df[metric].values

        c[i, :, 0] = engagement.astype(float)

    # Normalize engagement to [0, 1] per tweet
    if normalize:
        for i in range(num_tweets):
            max_val = c[i, :, 0].max()
            if max_val > 0:
                c[i, :, 0] = c[i, :, 0] / max_val
            # Tweets with 0 engagement throughout stay at 0

    print(f"Data shape - t: {t.shape}, c: {c.shape}")
    print(f"Time range: [{t.min():.3f}, {t.max():.3f}]")
    print(f"Engagement range: [{c.min():.3f}, {c.max():.3f}]")

    return t, c, post_ids


def inspect_data(snapshots_path):
    """
    Print summary statistics about the dataset.
    """
    df = pd.read_csv(snapshots_path)

    print("=== Dataset Summary ===")
    print(f"Total rows: {len(df)}")
    print(f"Unique tweets: {df['post_id'].nunique()}")
    print(f"Snapshots per tweet: {len(df) // df['post_id'].nunique()}")
    print()
    print("Engagement statistics:")
    for col in ['likes', 'retweets', 'replies', 'quotes']:
        print(f"  {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")
    print()

    # Check for tweets with actual growth (not flat)
    growth_tweets = 0
    for pid in df['post_id'].unique():
        tweet_df = df[df['post_id'] == pid]
        if tweet_df['likes'].max() > tweet_df['likes'].min():
            growth_tweets += 1
    print(f"Tweets with likes growth: {growth_tweets}/{df['post_id'].nunique()}")


def load_follower_counts(
    posts_path: str,
    authors_path: str,
    post_ids: List[str],
    verbose: bool = True
) -> np.ndarray:
    """
    Load follower counts for a list of post IDs by joining posts → authors.
    Returns log-transformed follower counts: log(followers_count + 1).

    Args:
        posts_path: Path to posts.csv (must have post_id, author_id)
        authors_path: Path to authors.csv (must have author_id, followers_count)
        post_ids: List of post IDs to get follower counts for
        verbose: If True, print statistics

    Returns:
        follower_log: (num_tweets,) array of log(followers_count + 1)
    """
    posts_df = pd.read_csv(posts_path, usecols=['post_id', 'author_id'])
    authors_df = pd.read_csv(authors_path, usecols=['author_id', 'followers_count'])

    # Join posts → authors
    merged = posts_df.merge(authors_df, on='author_id', how='left')

    # Build lookup: post_id → followers_count
    lookup = merged.set_index('post_id')['followers_count'].to_dict()

    # Get follower counts in the same order as post_ids
    follower_counts = np.array([lookup.get(pid, np.nan) for pid in post_ids])

    # Check for missing
    n_missing = np.isnan(follower_counts).sum()
    if n_missing > 0:
        raise ValueError(
            f"{n_missing}/{len(post_ids)} tweets have no follower data. "
            f"Check that posts.csv and authors.csv cover all post IDs."
        )

    # Log-transform: log(N + 1)
    # Follower counts range from 1 to ~36M; log compresses to ~0.7–17.4
    follower_log = np.log(follower_counts + 1)

    if verbose:
        print(f"\nFollower count statistics:")
        print(f"  Raw: min={follower_counts.min():.0f}, "
              f"max={follower_counts.max():.0f}, "
              f"median={np.median(follower_counts):.0f}")
        print(f"  Log-transformed: min={follower_log.min():.2f}, "
              f"max={follower_log.max():.2f}, "
              f"median={np.median(follower_log):.2f}")

    return follower_log


# Test the loader
if __name__ == '__main__':
    snapshots_path = '/Users/alexkoong/Desktop/Thesis/jan31 thesis final data download/snapshots.csv'

    # Inspect first
    inspect_data(snapshots_path)
    print()

    # Test original loader (for backward compatibility)
    print("=" * 60)
    print("Testing original loader (load_tweet_data):")
    print("=" * 60)
    t, c, post_ids = load_tweet_data(snapshots_path, metric='likes')

    # Show a few examples
    print("\nFirst 3 tweets (likes over time):")
    for i in range(min(3, len(post_ids))):
        print(f"  Tweet {post_ids[i]}: {c[i, :5, 0]} ... {c[i, -3:, 0]}")

    # Test robust loader
    print("\n" + "=" * 60)
    print("Testing robust loader (load_tweet_data_robust):")
    print("=" * 60)
    t_robust, c_robust, post_ids_robust, stats = load_tweet_data_robust(
        snapshots_path,
        metric='likes',
        start_hour=0,
        end_hour=47,
        max_gap_to_interpolate=2,
        normalize=True,
        verbose=True
    )

    # Show a few examples
    print("\nFirst 3 kept tweets (likes over time):")
    for i in range(min(3, len(post_ids_robust))):
        print(f"  Tweet {post_ids_robust[i]}: {c_robust[i, :5, 0]} ... {c_robust[i, -3:, 0]}")

    # Show interpolation details if any
    if stats.interpolation_details:
        print(f"\nTweets with interpolated values (first 5):")
        for i, (pid, hours) in enumerate(list(stats.interpolation_details.items())[:5]):
            print(f"  {pid}: interpolated hours {hours}")
