"""Command line:  watchlab parse | demo | run"""
import argparse
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="watchlab", description="Turn an Apple Health export into a readiness score and weekend advice.")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("parse", help="export.xml -> watch-only CSVs"); a.add_argument("export_xml"); a.add_argument("out_dir")
    b = sub.add_parser("demo", help="make fake data and run everything on it"); b.add_argument("--out", default="demo_data")
    c = sub.add_parser("run", help="run the pipeline on watch CSVs + a calendar_tags.csv")
    c.add_argument("watch_dir"); c.add_argument("calendar_csv"); c.add_argument("--out", default="outputs_run")
    args = p.parse_args(argv)
    if args.cmd == "parse":
        from parse_health import parse
        print(parse(args.export_xml, args.out_dir))
    elif args.cmd == "demo":
        from synthetic import make
        from pipeline import run_all
        make(f"{args.out}/watch")
        res = run_all(f"{args.out}/watch", f"{args.out}/calendar_tags.csv", f"{args.out}/outputs")
        print(open(f"{args.out}/outputs/weekend_plan.md").read())
    else:
        from pipeline import run_all
        run_all(args.watch_dir, args.calendar_csv, args.out)
        print(open(f"{args.out}/weekend_plan.md").read())


if __name__ == "__main__":
    sys.exit(main())
