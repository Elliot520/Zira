#!/bin/bash
# One-time setup ON YOUR SERVER (as root) for Zira's encrypted backups. Creates a limited account "zirabackup"
# that can only use SFTP inside /srv/zira-backups (no shell, no forwarding, no access to anything else), and
# trusts only the Mac's backup key. Touches nothing else on the server. Undo: see the end of this file.
#
#   Run it on the server:   sudo bash backup_server_setup.sh
set -euo pipefail

KEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKdApIK22RZiMTdmMmSH5ePp6vI1+yQVEH0Fno8bcBU1 zira-backup@Rehans-MacBook-Air'
BASE=/srv/zira-backups

id zirabackup >/dev/null 2>&1 || useradd --system --no-create-home --home-dir "$BASE" --shell /usr/sbin/nologin zirabackup
install -d -o root -g root -m 755 "$BASE"              # the chroot must be owned by root
install -d -o zirabackup -g zirabackup -m 700 "$BASE/repo"
install -d -o root -g root -m 755 "$BASE/.ssh"
printf 'restrict %s\n' "$KEY" > "$BASE/.ssh/authorized_keys"
chmod 644 "$BASE/.ssh/authorized_keys"

cat > /etc/ssh/sshd_config.d/zira-backup.conf <<'EOF'
# Zira backups: SFTP only, locked inside /srv/zira-backups.
Match User zirabackup
    ChrootDirectory /srv/zira-backups
    ForceCommand internal-sftp -d /repo
    AuthorizedKeysFile /srv/zira-backups/.ssh/authorized_keys
    PasswordAuthentication no
    AllowTcpForwarding no
    X11Forwarding no
    PermitTTY no
EOF

grep -qE '^\s*Include\s+/etc/ssh/sshd_config\.d/\*\.conf' /etc/ssh/sshd_config || {
    echo "Your sshd_config has no 'Include /etc/ssh/sshd_config.d/*.conf' line; add the Match block by hand."; exit 1; }
sshd -t
systemctl reload ssh 2>/dev/null || systemctl reload sshd
echo "Done. Free space for backups:"; df -h "$BASE" | tail -1

# Undo:  rm /etc/ssh/sshd_config.d/zira-backup.conf && systemctl reload ssh && userdel zirabackup && rm -rf /srv/zira-backups
