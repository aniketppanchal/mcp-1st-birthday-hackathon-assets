#!/bin/sh

/usr/local/bin/sandbox-api &

echo "127.0.0.1 localhost" >> /etc/hosts

wait
