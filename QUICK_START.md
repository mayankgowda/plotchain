# Quick Start Guide

## Running Tolerance Analysis Separately

You can now run the tolerance table generation independently:

```bash
python3 generate_tolerance_table.py --output_dir analysis_output
```

This will generate:
- `analysis_output/tables/tolerance_table.csv`
- `analysis_output/tables/tolerance_table.tex` (for paper appendix)

**Note**: The script automatically imports tolerance values from `run_plotchain_eval.py`, so no evaluation results are needed.

---

## Making Code Public - Summary

I've created a comprehensive checklist in `PUBLIC_RELEASE_CHECKLIST.md`. Here's what's been done and what remains:

### ✅ Already Completed

1. **Standalone tolerance script** - `generate_tolerance_table.py` created
2. **API keys** - Already using environment variables (safe)
3. **Debug logging** - Made conditional (only runs if `DEBUG_GEMINI=1` env var is set)
4. **.gitignore** - Updated to exclude `.env` files and debug logs
5. **.env.example** - Created template for API keys

### 📋 Remaining Tasks (Before Public Release)

#### High Priority
1. **Update README.md** - Current README is outdated:
   - Update dataset name and paths
   - Add installation instructions
   - Add usage examples
   - Add citation information

2. **Add LICENSE file** - Choose a license (MIT, Apache 2.0, etc.)

3. **Decide on data hosting**:
   - Include `data/plotchain/` in repo? (if < 100MB)
   - Or host separately and add download instructions

4. **Test installation** - Test on clean environment:
   ```bash
   # In fresh directory
   git clone <repo>
   cd plotchain_v1
   pip install -r requirements.txt
   pip install -r requirements_analysis.txt
   python3 generate_tolerance_table.py  # Should work
   ```

#### Medium Priority
5. **Review `paper_notes/`** - Decide if these should be:
   - Kept as-is (helpful for users)
   - Moved to `docs/paper/`
   - Removed

6. **Decide on `results/` and `analysis_output/`**:
   - Include for reproducibility?
   - Or exclude and let users regenerate?

### 📝 Files Created/Modified

**New Files**:
- `generate_tolerance_table.py` - Standalone tolerance table generator
- `PUBLIC_RELEASE_CHECKLIST.md` - Comprehensive checklist
- `.env.example` - API key template
- `QUICK_START.md` - This file

**Modified Files**:
- `run_plotchain_eval.py` - Debug logging now conditional
- `.gitignore` - Added `.env` and debug file patterns

### 🔍 Security Check

✅ **All API keys use environment variables** - Safe for public release
✅ **No hardcoded secrets** - Verified
✅ **Debug files excluded** - Added to `.gitignore`

### 🚀 Next Steps

1. Review `PUBLIC_RELEASE_CHECKLIST.md` for complete details
2. Update `README.md` with current information
3. Add `LICENSE` file
4. Test installation on clean environment
5. Decide on data/results hosting strategy
6. Create GitHub repository and push

---

## Testing the Tolerance Script

```bash
# Make sure you're in the project directory
cd /Users/mayankgowda/project/plotchain_v1

# Run the tolerance table generator
python3 generate_tolerance_table.py --output_dir analysis_output

# Check outputs
ls -lh analysis_output/tables/tolerance_table.*
```

The script should work immediately - it only needs `run_plotchain_eval.py` to exist (which it does).
