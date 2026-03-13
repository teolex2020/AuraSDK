//! Temporal SDR Encoding
//!
//! Resolves time-relative expressions into concrete SDR representations.
//!
//! # Math
//! ```text
//! T_resolved = f(T_relative, context_timestamp)
//! SDR_range = ∪{SDR(day) | day ∈ date_range}
//! ```
//!
//! # Examples
//! - "yesterday" → "2026-02-05"
//! - "last week" → union of 7 day SDRs
//! - "next Monday" → "2026-02-09"

use chrono::{Datelike, Duration, Local, NaiveDate, Weekday};
use std::collections::HashMap;

/// Temporal expression resolver
pub struct TemporalResolver {
    /// Cached relative expressions
    patterns: HashMap<String, TemporalPattern>,
}

/// Pattern type for temporal expressions
#[derive(Clone, Debug)]
#[allow(dead_code)]
enum TemporalPattern {
    /// Fixed offset from today (e.g., "yesterday" = -1)
    DayOffset(i64),
    /// Range of days (e.g., "last week" = [-7, -1])
    DayRange(i64, i64),
    /// Specific weekday in past (e.g., "last Monday")
    LastWeekday(Weekday),
    /// Specific weekday in future (e.g., "next Friday")
    NextWeekday(Weekday),
    /// Time of day (e.g., "morning", "evening")
    TimeOfDay(String),
    /// Month reference
    Month(u32),
}

impl TemporalResolver {
    /// Create a new temporal resolver with default patterns
    pub fn new() -> Self {
        let mut resolver = Self {
            patterns: HashMap::new(),
        };
        resolver.load_default_patterns();
        resolver
    }

    /// Load default temporal patterns
    fn load_default_patterns(&mut self) {
        use TemporalPattern::*;

        // Day offsets
        self.patterns.insert("today".to_string(), DayOffset(0));
        self.patterns.insert("now".to_string(), DayOffset(0));
        self.patterns.insert("yesterday".to_string(), DayOffset(-1));
        self.patterns.insert("tomorrow".to_string(), DayOffset(1));
        self.patterns
            .insert("day before yesterday".to_string(), DayOffset(-2));
        self.patterns
            .insert("day after tomorrow".to_string(), DayOffset(2));

        // Ranges
        self.patterns
            .insert("last week".to_string(), DayRange(-7, -1));
        self.patterns
            .insert("this week".to_string(), DayRange(-6, 0));
        self.patterns
            .insert("next week".to_string(), DayRange(1, 7));
        self.patterns
            .insert("last month".to_string(), DayRange(-30, -1));
        self.patterns
            .insert("this month".to_string(), DayRange(-29, 0));
        self.patterns
            .insert("last year".to_string(), DayRange(-365, -1));
        self.patterns
            .insert("past week".to_string(), DayRange(-7, 0));
        self.patterns
            .insert("past month".to_string(), DayRange(-30, 0));
        self.patterns
            .insert("recently".to_string(), DayRange(-7, 0));
        self.patterns.insert("lately".to_string(), DayRange(-14, 0));

        // Weekdays (last)
        self.patterns
            .insert("last monday".to_string(), LastWeekday(Weekday::Mon));
        self.patterns
            .insert("last tuesday".to_string(), LastWeekday(Weekday::Tue));
        self.patterns
            .insert("last wednesday".to_string(), LastWeekday(Weekday::Wed));
        self.patterns
            .insert("last thursday".to_string(), LastWeekday(Weekday::Thu));
        self.patterns
            .insert("last friday".to_string(), LastWeekday(Weekday::Fri));
        self.patterns
            .insert("last saturday".to_string(), LastWeekday(Weekday::Sat));
        self.patterns
            .insert("last sunday".to_string(), LastWeekday(Weekday::Sun));

        // Weekdays (next)
        self.patterns
            .insert("next monday".to_string(), NextWeekday(Weekday::Mon));
        self.patterns
            .insert("next tuesday".to_string(), NextWeekday(Weekday::Tue));
        self.patterns
            .insert("next wednesday".to_string(), NextWeekday(Weekday::Wed));
        self.patterns
            .insert("next thursday".to_string(), NextWeekday(Weekday::Thu));
        self.patterns
            .insert("next friday".to_string(), NextWeekday(Weekday::Fri));
        self.patterns
            .insert("next saturday".to_string(), NextWeekday(Weekday::Sat));
        self.patterns
            .insert("next sunday".to_string(), NextWeekday(Weekday::Sun));

        // Time of day
        self.patterns
            .insert("morning".to_string(), TimeOfDay("morning".to_string()));
        self.patterns
            .insert("this morning".to_string(), TimeOfDay("morning".to_string()));
        self.patterns
            .insert("afternoon".to_string(), TimeOfDay("afternoon".to_string()));
        self.patterns.insert(
            "this afternoon".to_string(),
            TimeOfDay("afternoon".to_string()),
        );
        self.patterns
            .insert("evening".to_string(), TimeOfDay("evening".to_string()));
        self.patterns
            .insert("this evening".to_string(), TimeOfDay("evening".to_string()));
        self.patterns
            .insert("tonight".to_string(), TimeOfDay("night".to_string()));
        self.patterns
            .insert("last night".to_string(), TimeOfDay("night".to_string()));
    }

    /// Resolve temporal expressions in text to concrete dates
    pub fn resolve(&self, text: &str) -> String {
        let text_lower = text.to_lowercase();
        let today = Local::now().date_naive();
        let mut result = text.to_string();

        // Check each pattern (longest first for proper matching)
        let mut patterns: Vec<_> = self.patterns.iter().collect();
        patterns.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        for (pattern, temporal) in patterns {
            if text_lower.contains(pattern) {
                let replacement = self.resolve_pattern(temporal, today);
                // Case-insensitive replacement
                let pattern_regex =
                    regex::Regex::new(&format!("(?i){}", regex::escape(pattern))).unwrap();
                result = pattern_regex
                    .replace_all(&result, replacement.as_str())
                    .to_string();
            }
        }

        result
    }

    /// Resolve a single pattern to date string(s)
    fn resolve_pattern(&self, pattern: &TemporalPattern, today: NaiveDate) -> String {
        match pattern {
            TemporalPattern::DayOffset(offset) => {
                let date = today + Duration::days(*offset);
                date.format("%Y-%m-%d").to_string()
            }
            TemporalPattern::DayRange(start, end) => {
                let start_date = today + Duration::days(*start);
                let end_date = today + Duration::days(*end);
                format!(
                    "{} to {}",
                    start_date.format("%Y-%m-%d"),
                    end_date.format("%Y-%m-%d")
                )
            }
            TemporalPattern::LastWeekday(weekday) => {
                let date = self.find_last_weekday(today, *weekday);
                date.format("%Y-%m-%d").to_string()
            }
            TemporalPattern::NextWeekday(weekday) => {
                let date = self.find_next_weekday(today, *weekday);
                date.format("%Y-%m-%d").to_string()
            }
            TemporalPattern::TimeOfDay(time) => {
                format!("{} {}", today.format("%Y-%m-%d"), time)
            }
            TemporalPattern::Month(month) => {
                let year = if *month > today.month() {
                    today.year() - 1
                } else {
                    today.year()
                };
                format!("{}-{:02}", year, month)
            }
        }
    }

    /// Find the last occurrence of a weekday
    fn find_last_weekday(&self, from: NaiveDate, weekday: Weekday) -> NaiveDate {
        let mut date = from - Duration::days(1);
        while date.weekday() != weekday {
            date -= Duration::days(1);
        }
        date
    }

    /// Find the next occurrence of a weekday
    fn find_next_weekday(&self, from: NaiveDate, weekday: Weekday) -> NaiveDate {
        let mut date = from + Duration::days(1);
        while date.weekday() != weekday {
            date += Duration::days(1);
        }
        date
    }

    /// Get all dates in a range for SDR expansion
    pub fn get_date_range(&self, text: &str) -> Vec<NaiveDate> {
        let text_lower = text.to_lowercase();
        let today = Local::now().date_naive();

        for (pattern, temporal) in &self.patterns {
            if text_lower.contains(pattern) {
                if let TemporalPattern::DayRange(start, end) = temporal {
                    let mut dates = Vec::new();
                    for offset in *start..=*end {
                        dates.push(today + Duration::days(offset));
                    }
                    return dates;
                }
            }
        }

        Vec::new()
    }

    /// Extract date from resolved text
    pub fn extract_date(&self, text: &str) -> Option<NaiveDate> {
        // Try to parse YYYY-MM-DD format
        let re = regex::Regex::new(r"(\d{4}-\d{2}-\d{2})").unwrap();
        if let Some(cap) = re.captures(text) {
            if let Ok(date) = NaiveDate::parse_from_str(&cap[1], "%Y-%m-%d") {
                return Some(date);
            }
        }
        None
    }

    /// Compute recency weight for a date (0.0 to 1.0)
    ///
    /// Math: weight = decay(now - timestamp)
    pub fn recency_weight(&self, date: NaiveDate, half_life_days: f64) -> f64 {
        let today = Local::now().date_naive();
        let days_ago = (today - date).num_days() as f64;

        if days_ago < 0.0 {
            // Future date
            1.0
        } else {
            // Exponential decay: w = 2^(-t/half_life)
            2.0_f64.powf(-days_ago / half_life_days)
        }
    }

    /// Get number of patterns
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
}

impl Default for TemporalResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yesterday_resolution() {
        let resolver = TemporalResolver::new();
        let result = resolver.resolve("What happened yesterday?");

        // Should contain a date in YYYY-MM-DD format
        assert!(result.contains("-"));
        assert!(!result.contains("yesterday"));
    }

    #[test]
    fn test_last_week_resolution() {
        let resolver = TemporalResolver::new();
        let result = resolver.resolve("Events from last week");

        // Should contain date range
        assert!(result.contains(" to "));
    }

    #[test]
    fn test_recency_weight() {
        let resolver = TemporalResolver::new();
        let today = Local::now().date_naive();

        let weight_today = resolver.recency_weight(today, 7.0);
        let weight_week_ago = resolver.recency_weight(today - Duration::days(7), 7.0);

        assert!((weight_today - 1.0).abs() < 0.01);
        assert!((weight_week_ago - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_date_range() {
        let resolver = TemporalResolver::new();
        let dates = resolver.get_date_range("last week");

        assert_eq!(dates.len(), 7);
    }

    #[test]
    fn test_weekday_resolution() {
        let resolver = TemporalResolver::new();
        let result = resolver.resolve("Meeting on last monday");

        assert!(result.contains("-"));
        assert!(!result.to_lowercase().contains("last monday"));
    }
}
