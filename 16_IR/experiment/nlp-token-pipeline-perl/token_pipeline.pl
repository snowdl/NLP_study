use strict;
use warnings;

my $text = <STDIN>;
die "No input provided.\n" unless defined $text;

chomp $text;
$text = lc $text;   # normalize

print "RAW: $text\n";
