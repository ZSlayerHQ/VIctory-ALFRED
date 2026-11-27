---
name: Task
about: A scoped task in one area, with a done-when
title: "[task] "
labels: ["task"]
---

## Area

One area: tools and measurement, CPU side (speed and memory), or GPU side (picture, shaders, the card). The
areas and the files each holds are in TASKS.md.

## Cards

`nvidia`, `amd`, `intel` or `all`: the graphics cards whose behaviour this task concerns. A task for one vendor
still needs a statement that the others do not regress (TASKS.md, "Cards").

## Goal

The task in two sentences: what is wrong or missing today, and what will be true when it is done.

## What is known, and how

What has been established and by what: an established tool's measurement (name the tool), a test, or reading the
code. A cause that is only suspected says "suspected".

## Where the code is

The files and functions this task may change. Give the crate and the path, and name the function or constant where
it helps. If the files are not certain yet, write "to be confirmed by a maintainer" and say what would confirm them.

## What done means

The exact commands and what they must show. A bug fix starts with a test that fails before the change and passes
after it; a diagnostic or a note says what it must contain. Where a run on a GPU is needed, name the run, the
switches set and the figures to paste, and the established tool that judges a speed or picture claim.

## The test the change needs

The fixture, the assertion, or the reason no test can exist; whether it runs without the title (on the software
adapter) or needs a run with your own copy. "The existing tests pass" is not a test for a new behaviour.

## Out of scope

What this task does not change, so the pull request stays in one task and one sitting.

## Size

S, M or L. One agent should finish it in one to three days.

## Hardware or files needed

Say "none" when the task can be done from this repository alone. Otherwise say what is needed: a copy of the title,
a Direct3D 12 GPU of a named vendor or tier, a second GPU to confirm a fix, or a capture from a run.

## Content check

- [ ] This task holds no part of the game: no captures, assets, executable bytes, dumps, bulk extracted tables, or
      paths into a game dump.

