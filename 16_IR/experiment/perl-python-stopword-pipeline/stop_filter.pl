use strict;
use warnings;

my ($stopfile, $infile) = @ARGV;
die "Usage: perl $0 stop.txt input.txt|-\n" unless $stopfile;

# 1) stopword load -> %stop
open my $sf, "<:encoding(UTF-8)", $stopfile or die "Cannot open $stopfile: $!";
my %stop;
while (<$sf>) {
  chomp;
  s/^\s+|\s+$//g;            # trim
  next if $_ eq "" || /^#/;  # skip blank/comment
  $stop{lc $_} = 1;
}
close $sf;

# 2) input handle (file or STDIN)
my $fh;
if (!defined $infile || $infile eq '-') {
  $fh = \*STDIN;  # ✅ 중요: 파이프 입력 받을 때 이게 더 안전
} else {
  open $fh, "<:encoding(UTF-8)", $infile or die "Cannot open $infile: $!";
}

while (<$fh>) {
  chomp;
  my @w = split /\s+/;
  my @out;

  for my $tok (@w) {
    $tok = lc $tok;
    $tok =~ s/^[^[:alnum:]]+|[^[:alnum:]]+$//g; # strip punct

    next if $tok eq "";
    next if exists $stop{$tok};
    push @out, $tok;
  }

  print join(" ", @out), "\n";
}

close $fh if defined $infile && $infile ne '-';