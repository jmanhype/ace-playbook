"""Ed25519 signing for BLACKICE 3.0 Enterprise receipts.

Provides cryptographic signing and verification using Ed25519:
- Key pair generation
- Receipt signing
- Signature verification
- Key serialization/deserialization
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nacl.encoding import HexEncoder
from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey as NaClSigningKey
from nacl.signing import VerifyKey

from blackice.primitives.types import Timestamp

if TYPE_CHECKING:
    from blackice.schemas.receipt import Receipt, Signature


class SigningError(Exception):
    """Error during signing operation."""

    pass


class VerificationError(Exception):
    """Error during signature verification."""

    pass


@dataclass
class KeyPair:
    """Ed25519 key pair for receipt signing.

    Attributes:
        key_id: Unique identifier for this key pair
        private_key: Base64-encoded private key (32 bytes seed)
        public_key: Base64-encoded public key (32 bytes)
        created_at: When the key was generated
    """

    key_id: str
    private_key: str  # Base64-encoded
    public_key: str  # Base64-encoded
    created_at: Timestamp

    def get_signing_key(self) -> NaClSigningKey:
        """Get the NaCl signing key for this key pair."""
        seed = base64.b64decode(self.private_key)
        return NaClSigningKey(seed)

    def get_verify_key(self) -> VerifyKey:
        """Get the NaCl verify key for this key pair."""
        public_bytes = base64.b64decode(self.public_key)
        return VerifyKey(public_bytes)

    @classmethod
    def from_file(cls, path: Path) -> KeyPair:
        """Load key pair from file.

        File format is INI-style:
        [key]
        id = key-id
        private = base64-encoded-private-key
        public = base64-encoded-public-key
        created = ISO-8601-timestamp
        """
        import configparser

        config = configparser.ConfigParser()
        config.read(path)

        return cls(
            key_id=config["key"]["id"],
            private_key=config["key"]["private"],
            public_key=config["key"]["public"],
            created_at=Timestamp(value=config["key"]["created"]),
        )

    def to_file(self, path: Path) -> None:
        """Save key pair to file."""
        import configparser

        config = configparser.ConfigParser()
        config["key"] = {
            "id": self.key_id,
            "private": self.private_key,
            "public": self.public_key,
            "created": str(self.created_at),
        }

        with open(path, "w") as f:
            config.write(f)


def generate_key_pair(key_id: str | None = None) -> KeyPair:
    """Generate a new Ed25519 key pair.

    Args:
        key_id: Optional key identifier. If not provided, generates one
                from the public key hash.

    Returns:
        New KeyPair with generated keys
    """
    # Generate random signing key
    signing_key = NaClSigningKey.generate()

    # Extract public key
    verify_key = signing_key.verify_key

    # Encode keys as base64
    private_key_b64 = base64.b64encode(bytes(signing_key)).decode("ascii")
    public_key_b64 = base64.b64encode(bytes(verify_key)).decode("ascii")

    # Generate key ID if not provided
    if key_id is None:
        key_hash = hashlib.sha256(bytes(verify_key)).hexdigest()[:16]
        key_id = f"blackice-{key_hash}"

    return KeyPair(
        key_id=key_id,
        private_key=private_key_b64,
        public_key=public_key_b64,
        created_at=Timestamp.now(),
    )


def load_private_key(private_key_b64: str) -> NaClSigningKey:
    """Load a signing key from base64-encoded private key.

    Args:
        private_key_b64: Base64-encoded 32-byte seed

    Returns:
        NaCl SigningKey for signing operations
    """
    try:
        seed = base64.b64decode(private_key_b64)
        return NaClSigningKey(seed)
    except Exception as e:
        raise SigningError(f"Invalid private key: {e}") from e


def load_public_key(public_key_b64: str) -> VerifyKey:
    """Load a verify key from base64-encoded public key.

    Args:
        public_key_b64: Base64-encoded 32-byte public key

    Returns:
        NaCl VerifyKey for verification operations
    """
    try:
        public_bytes = base64.b64decode(public_key_b64)
        return VerifyKey(public_bytes)
    except Exception as e:
        raise VerificationError(f"Invalid public key: {e}") from e


def sign_receipt(receipt: Receipt, key_pair: KeyPair) -> Signature:
    """Sign a receipt with the given key pair.

    Creates an Ed25519 signature of the receipt's digest.

    Args:
        receipt: The receipt to sign
        key_pair: Key pair to use for signing

    Returns:
        Signature object containing the signature and metadata
    """
    from blackice.schemas.receipt import Signature

    # Get the digest to sign
    digest = receipt.compute_digest()
    digest_bytes = bytes.fromhex(digest)

    # Sign with Ed25519
    signing_key = key_pair.get_signing_key()
    signed = signing_key.sign(digest_bytes)

    # Extract just the signature (first 64 bytes, message is appended)
    signature_bytes = signed.signature

    # Encode signature as base64
    signature_b64 = base64.b64encode(signature_bytes).decode("ascii")

    return Signature(
        algorithm="ed25519",
        public_key_id=key_pair.key_id,
        signature=signature_b64,
        timestamp=Timestamp.now(),
    )


def verify_receipt_signature(
    receipt: Receipt,
    signature: Signature,
    public_key: str | VerifyKey,
) -> tuple[bool, str]:
    """Verify a receipt's signature.

    Args:
        receipt: The receipt that was signed
        signature: The signature to verify
        public_key: Base64-encoded public key or VerifyKey object

    Returns:
        (is_valid, message) tuple
    """
    # Ensure we have a VerifyKey
    if isinstance(public_key, str):
        try:
            verify_key = load_public_key(public_key)
        except VerificationError as e:
            return False, str(e)
    else:
        verify_key = public_key

    # Verify algorithm
    if signature.algorithm != "ed25519":
        return False, f"Unsupported algorithm: {signature.algorithm}"

    # Get the digest that was signed
    digest = receipt.compute_digest()
    digest_bytes = bytes.fromhex(digest)

    # Decode the signature
    try:
        signature_bytes = base64.b64decode(signature.signature)
    except Exception as e:
        return False, f"Invalid signature encoding: {e}"

    # Verify the signature
    try:
        verify_key.verify(digest_bytes, signature_bytes)
        return True, "Signature verified successfully"
    except BadSignatureError:
        return False, "Signature verification failed: invalid signature"
    except Exception as e:
        return False, f"Signature verification failed: {e}"


def sign_data(data: bytes, key_pair: KeyPair) -> str:
    """Sign arbitrary data with the given key pair.

    Args:
        data: Data to sign
        key_pair: Key pair to use for signing

    Returns:
        Base64-encoded signature
    """
    signing_key = key_pair.get_signing_key()
    signed = signing_key.sign(data)
    return base64.b64encode(signed.signature).decode("ascii")


def verify_data(
    data: bytes,
    signature_b64: str,
    public_key: str | VerifyKey,
) -> tuple[bool, str]:
    """Verify a signature on arbitrary data.

    Args:
        data: Data that was signed
        signature_b64: Base64-encoded signature
        public_key: Base64-encoded public key or VerifyKey object

    Returns:
        (is_valid, message) tuple
    """
    # Ensure we have a VerifyKey
    if isinstance(public_key, str):
        try:
            verify_key = load_public_key(public_key)
        except VerificationError as e:
            return False, str(e)
    else:
        verify_key = public_key

    # Decode the signature
    try:
        signature_bytes = base64.b64decode(signature_b64)
    except Exception as e:
        return False, f"Invalid signature encoding: {e}"

    # Verify the signature
    try:
        verify_key.verify(data, signature_bytes)
        return True, "Signature verified successfully"
    except BadSignatureError:
        return False, "Signature verification failed: invalid signature"
    except Exception as e:
        return False, f"Signature verification failed: {e}"
