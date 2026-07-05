//! vLLM tag ordering + `git ls-remote` parsing for the release watcher.
//!
//! Pure and heavily tested: the workflow does the network (resolves the latest
//! release and dumps every tag), and this module decides what to pin. Keeping
//! the version math here (not in brittle shell `sort -V`) is what lets us order
//! release candidates correctly (`0.24.0rc1 < 0.24.0`).

use std::cmp::Ordering;
use std::collections::BTreeMap;

/// A parsed vLLM tag: `vMAJOR.MINOR.PATCH` with an optional `rcN` pre-release.
///
/// Ordering is semver-ish: an rc sorts *before* its own final release
/// (`0.24.0rc1 < 0.24.0`) but after the previous release (`0.23.0 < 0.24.0rc1`),
/// and rc numbers compare numerically (`rc1 < rc2 < rc10`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VllmVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    /// `None` = final release; `Some(n)` = the `rcN` pre-release.
    pub rc: Option<u32>,
}

impl VllmVersion {
    /// Parse `v0.24.0`, `0.24.0`, or `v0.24.0rc1`. Returns `None` for anything
    /// that isn't `MAJOR.MINOR.PATCH[rcN]` (dev builds, other pre-release
    /// flavors, junk) so unknown tag shapes are skipped rather than mis-ordered.
    pub fn parse(tag: &str) -> Option<Self> {
        let core = tag.trim().trim_start_matches('v');
        let mut parts = core.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch_field = parts.next()?;
        if parts.next().is_some() {
            return None; // more than three dotted components
        }
        let (patch_str, rc) = match patch_field.split_once("rc") {
            Some((patch, n)) => (patch, Some(n.parse().ok()?)),
            None => (patch_field, None),
        };
        Some(Self {
            major,
            minor,
            patch: patch_str.parse().ok()?,
            rc,
        })
    }

    /// The `MAJOR.MINOR` line label, e.g. `"0.24"` (matches `sim_compat::minor_line`).
    pub fn minor_line(&self) -> String {
        format!("{}.{}", self.major, self.minor)
    }
}

impl Ord for VllmVersion {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.major, self.minor, self.patch)
            .cmp(&(other.major, other.minor, other.patch))
            .then_with(|| match (self.rc, other.rc) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Greater, // final release > its rc
                (Some(_), None) => Ordering::Less,
                (Some(a), Some(b)) => a.cmp(&b),
            })
    }
}

impl PartialOrd for VllmVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Parse `git ls-remote --tags` output into `tag -> commit sha`.
///
/// Each line is `<sha>\t<ref>`. Annotated tags emit two lines: the tag object
/// (`refs/tags/v1`) and the commit it points at (`refs/tags/v1^{}`). We want the
/// commit, so a `^{}` deref always wins over the bare tag-object sha.
pub fn parse_ls_remote(text: &str) -> BTreeMap<String, String> {
    let mut tags = BTreeMap::new();
    for line in text.lines() {
        let Some((sha, refname)) = line.split_once('\t') else {
            continue;
        };
        let Some(tag) = refname.trim().strip_prefix("refs/tags/") else {
            continue;
        };
        match tag.strip_suffix("^{}") {
            // Deref line: the real commit, overrides the tag-object entry.
            Some(name) => {
                tags.insert(name.to_string(), sha.trim().to_string());
            }
            // Bare tag: keep only if we haven't already got (and prefer) a deref.
            None => {
                tags.entry(tag.to_string())
                    .or_insert_with(|| sha.trim().to_string());
            }
        }
    }
    tags
}

/// The newest release-candidate tag strictly newer than `stable`, with its sha.
///
/// "Newer than stable" filters out leftover rc tags for an already-released line
/// (`0.24.0rc2` once `0.24.0` is out: the rc sorts *below* the release, so it's
/// dropped). Returns `None` when no such rc exists, i.e. there is nothing
/// unreleased to track yet.
pub fn pick_latest_rc(
    tags: &BTreeMap<String, String>,
    stable: VllmVersion,
) -> Option<(String, String)> {
    tags.iter()
        .filter_map(|(name, sha)| Some((VllmVersion::parse(name)?, name, sha)))
        .filter(|(v, _, _)| v.rc.is_some() && *v > stable)
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, name, sha)| (name.clone(), sha.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> VllmVersion {
        VllmVersion::parse(s).unwrap_or_else(|| panic!("parse {s}"))
    }

    #[test]
    fn parses_release_and_rc_tags() {
        assert_eq!(
            v("v0.24.0"),
            VllmVersion {
                major: 0,
                minor: 24,
                patch: 0,
                rc: None
            }
        );
        assert_eq!(
            v("0.24.0rc1"),
            VllmVersion {
                major: 0,
                minor: 24,
                patch: 0,
                rc: Some(1)
            }
        );
        assert_eq!(v("v0.23.1rc0").rc, Some(0));
        assert_eq!(v("v0.24.0rc1").minor_line(), "0.24");
    }

    #[test]
    fn rejects_non_release_shapes() {
        for junk in [
            "nightly",
            "",
            "v0.24",
            "0.24.0.dev1+g16e9",
            "v0.24.0a1",
            "garbage",
        ] {
            assert!(VllmVersion::parse(junk).is_none(), "should reject {junk}");
        }
    }

    #[test]
    fn rc_sorts_below_its_release_and_above_the_previous() {
        assert!(v("v0.24.0rc1") < v("v0.24.0")); // rc precedes its own release
        assert!(v("v0.23.0") < v("v0.24.0rc1")); // but follows the prior release
        assert!(v("v0.24.0rc1") < v("v0.24.0rc2")); // rc numbers are numeric
        assert!(v("v0.24.0rc2") < v("v0.24.0rc10")); // not lexicographic
        assert!(v("v0.23.1rc0") < v("v0.24.0rc1")); // patch line bumps win
        assert!(v("v0.23.0") < v("v0.23.1rc0")); // a patch rc is newer than the .0
    }

    #[test]
    fn ls_remote_prefers_dereferenced_commit() {
        let text = "\
aaaa1111\trefs/tags/v0.24.0rc1
bbbb2222\trefs/tags/v0.24.0rc1^{}
cccc3333\trefs/tags/v0.23.0
dddd4444\trefs/heads/main
not-a-tab-line
";
        let tags = parse_ls_remote(text);
        assert_eq!(tags.get("v0.24.0rc1").map(String::as_str), Some("bbbb2222"));
        assert_eq!(tags.get("v0.23.0").map(String::as_str), Some("cccc3333"));
        assert!(!tags.contains_key("main"), "branches are not tags");
    }

    fn tagset(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(n, s)| (n.to_string(), s.to_string()))
            .collect()
    }

    #[test]
    fn picks_highest_rc_newer_than_stable() {
        let tags = tagset(&[
            ("v0.24.0rc1", "sha-2400rc1"),
            ("v0.23.1rc0", "sha-2310rc0"),
            ("v0.23.0", "sha-2300"),
            ("v0.22.1", "sha-2210"),
        ]);
        let (tag, sha) = pick_latest_rc(&tags, v("v0.23.0")).expect("an rc exists");
        assert_eq!(tag, "v0.24.0rc1");
        assert_eq!(sha, "sha-2400rc1");
    }

    #[test]
    fn ignores_rc_for_already_released_line() {
        // 0.24.0 is out; only a stale 0.24.0rc2 tag remains -> nothing to track.
        let tags = tagset(&[("v0.24.0", "sha-rel"), ("v0.24.0rc2", "sha-rc2")]);
        assert!(pick_latest_rc(&tags, v("v0.24.0")).is_none());
    }

    #[test]
    fn no_rc_when_none_newer_than_stable() {
        let tags = tagset(&[("v0.23.0", "sha"), ("v0.22.1", "sha")]);
        assert!(pick_latest_rc(&tags, v("v0.23.0")).is_none());
    }
}
