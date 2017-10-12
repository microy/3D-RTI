#! /bin/sh

find . -type f -name '*.png' -not -path "./Images/*" -print -delete

