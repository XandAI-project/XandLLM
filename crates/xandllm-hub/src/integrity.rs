use std::path::Path;

use sha2::{Digest, Sha256};
use tracing::debug;

use crate::error::{HubError, HubResult};

/// Compute the SHA-256 hex digest of a file on disk.
pub fn sha256_file(path: &Path) -> HubResult<String> {
    let data = std::fs::read(path).map_err(HubError::Io)?;
    let digest = Sha256::digest(&data);
    Ok(hex::encode(digest))
}

/// Verify that `path` matches the expected SHA-256 hex string.
///
/// Returns `Ok(())` on success, `Err(HubError::IntegrityMismatch)` on failure.
pub fn verify_integrity(path: &Path, expected_sha256: &str) -> HubResult<()> {
    let actual = sha256_file(path)?;
    debug!(
        path = %path.display(),
        expected = expected_sha256,
        actual = %actual,
        "Verifying file integrity"
    );
    if actual.eq_ignore_ascii_case(expected_sha256) {
        Ok(())
    } else {
        Err(HubError::IntegrityMismatch {
            file: path.to_string_lossy().to_string(),
            expected: expected_sha256.to_string(),
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn write_temp_file(name: &str, contents: &[u8]) -> std::path::PathBuf {
        let path = temp_dir().join(name);
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn test_sha256_known_value() {
        // SHA-256 of the ASCII string "hello" is well-known
        let path = write_temp_file("xandllm_sha_test.txt", b"hello");
        let digest = sha256_file(&path).unwrap();
        assert_eq!(
            digest,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_verify_integrity_matching() {
        let path = write_temp_file("xandllm_verify_ok.txt", b"hello");
        let result = verify_integrity(
            &path,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
        );
        assert!(result.is_ok(), "matching hash should pass: {result:?}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_verify_integrity_case_insensitive() {
        let path = write_temp_file("xandllm_verify_case.txt", b"hello");
        let result = verify_integrity(
            &path,
            "2CF24DBA5FB0A30E26E83B2AC5B9E29E1B161E5C1FA7425E73043362938B9824",
        );
        assert!(result.is_ok(), "hash comparison should be case-insensitive");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_verify_integrity_mismatch() {
        let path = write_temp_file("xandllm_verify_fail.txt", b"hello");
        let result = verify_integrity(&path, "0000000000000000000000000000000000000000000000000000000000000000");
        assert!(
            matches!(result, Err(HubError::IntegrityMismatch { .. })),
            "wrong hash should return IntegrityMismatch"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_sha256_empty_file() {
        let path = write_temp_file("xandllm_sha_empty.txt", b"");
        let digest = sha256_file(&path).unwrap();
        // SHA-256 of empty string
        assert_eq!(
            digest,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        std::fs::remove_file(&path).ok();
    }
}
